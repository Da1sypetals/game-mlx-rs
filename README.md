```
--test-threads=1
```
is required for tests.

```
MLX 使用 Metal GPU，其底层状态是进程级全局的（一个 MTLDevice，一个命令队列）。Rust 测试框架默认多线程并发跑所有测试，多个测试同时调用 MLX 推理 → 并发提交 Metal 命令 → Metal 驱动层竞争条件 → SIGSEGV。

具体来说：

transcribe_returns_notes 和 note_times_are_monotonic 都会调用 model.infer()，后者内部会 mlx_rs::transforms::eval(...) 触发 Metal kernel dispatch
两个线程同时 dispatch 到同一个 MTLCommandQueue 而 MLX 内部没有做线程安全保护
结果是访问了已经被另一个线程释放或重用的 GPU buffer → invalid memory reference
--test-threads=1 让测试串行运行，Metal 命令队列按顺序提交，不再竞争，所以不崩。
```