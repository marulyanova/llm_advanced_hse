### **Тестовые запуски, эксперименты с параметрами:**

| run                                                                         | comment                                                                                                                                                                                                                                                                                                                                                               | tokens_<br>per_<br>second_<br>including_<br>warmup | tokens_<br>per_<br>second | peak_<br>allocated_<br>gib | peak_<br>reserved_<br>gib | train_<br>seconds | optimizer_<br>steps | target_<br>tokens | warmup_<br>steps | measured_<br>steps | measured_<br>seconds | seconds_<br>per_<br>optimizer_<br>step | validation_<br>loss | validation_<br>target_<br>tokens |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------- | -------------------------- | ------------------------- | ----------------- | ------------------- | ----------------- | ---------------- | ------------------ | -------------------- | -------------------------------------- | ------------------- | -------------------------------- |
| baseline                                                                    | -                                                                                                                                                                                                                                                                                                                                                                     | 7002.9708                                          | 7052.2466                 | 41.2370                    | 43.4395                   | 300.8527          | 370                 | 2106863           | 10               | 360                | 290.5995             | 0.8072                                 | 6.5938              | 50694                            |
| batch_size16                                                                | Уменьшаем batch size с 32 из бейзлайна до 16. Token per second уже возрасло                                                                                                                                                                                                                                                                                           | 7745.3598                                          | 7988.6317                 | 24.2110                    | 25.9961                   | 60.0685           | 164                 | 465252            | 10               | 154                | 54.3895              | 0.3532                                 | 8.0504              | 50694                            |
| batch_size64                                                                | Переполнение памяти CUDA. Эксперимент не завершился                                                                                                                                                                                                                                                                                                                   |                                                    |                           |                            |                           |                   |                     |                   |                  |                    |                      |                                        |                     |                                  |
| batch_size16_<br>gradacc8                                                   | Поднимаем аккумулирование градиента вместо бейзлайновых 2 - 8. Token per second ещё возрасло                                                                                                                                                                                                                                                                          | 7896.7056                                          | 8291.3704                 | 24.2114                    | 25.9961                   | 60.4774           | 42                  | 477572            | 10               | 32                 | 44.0698              | 1.3772                                 | 8.9478              | 50694                            |
| batch_size16_<br>gradacc16                                                  | Ещё поднимаем аккумулирование градиента вместо бейзлайновых 2 - 16. Token per second ещё возрасло                                                                                                                                                                                                                                                                     | 7921.9176                                          | 8429.4490                 | 24.2158                    | 26.2246                   | 60.2849           | 21                  | 477572            | 10               | 11                 | 30.5223              | 2.7748                                 | 10.8660             | 50694                            |
| batch_size16_<br>gradacc16_<br>gradcheckTrue                                | Попробуем добавить gradient checkpointing. Что делает: не сохраняет часть промежуточных активаций и пересчитывает их при backward. Видим замедление, что ожидаемо (token per second просел хуже бейзлайна), но расходуется памяти в 3 раза меньше, чем бейзлайн и в 2 раза меньше, чем предыдущий ран (peak_allocated_gib)                                            | 6449.8241                                          | 6858.1977                 | 12.2961                    | 19.1055                   | 62.9495           | 18                  | 406013            | 10               | 8                  | 27.0810              | 3.3851                                 | 11.0237             | 50694                            |
| batch_size16_<br>gradacc16_<br>paddingfree                                  | Попробуем метод, который не использует концепцию паддингов. Склеивает примеры, объединяет примеры вдоль оси последовательности без добавления паддинга, а тексты разделяются между собой при помощи маскирования. Очень сильный прирост по эффективности token per second, а памяти тратится также мало, как и с gradient checkpointing                               | 15987.4788                                         | 16289.0819                | 11.9275                    | 13.4980                   | 61.1134           | 43                  | 977049            | 10               | 33                 | 46.4583              | 1.4078                                 | 8.7595              | 50694                            |
| batch_size16_<br>gradacc16_<br>paddingfree_<br>flashat_<br>actoffload       | Пробуем поменять дефолтное SDPA на Flash Attention и сделать activation offloading - это техника экстремальной экономии видеопамяти - активации, которые обычно хранятся в быстрой, но маленькой видеопамяти GPU, принудительно сохраняются на CPU.<br>Это должно было еще снизить peak_allocated_gib и peak_reserved_gib, но эксперимент не запустился. См. ниже лог |                                                    |                           |                            |                           |                   |                     |                   |                  |                    |                      |                                        |                     |                                  |
| batch_size16_<br>gradacc16_<br>paddingfree_<br>flashatt                     | Без activation offloading только с flash attention запустилось, результат лучше по token per second, чем у SDPA. По памяти не увеличилось                                                                                                                                                                                                                             | 17551.5048                                         | 17926.6508                | 11.7109                    | 13.1445                   | 60.7080           | 47                  | 1065516           | 10               | 37                 | 47.1494              | 1.2743                                 | 8.5315              | 50694                            |
| batch_size16_<br>gradacc16_<br>paddingfree_<br>flashatt_<br>compile         | Попробуем с torch.compile скомпилировать. Не удалось. См. логи ниже                                                                                                                                                                                                                                                                                                   |                                                    |                           |                            |                           |                   |                     |                   |                  |                    |                      |                                        |                     |                                  |
| batch_size16_<br>gradacc16_<br>paddingfree_<br>flashatt_<br>compileinductor | Попробуем по другому скомпилировать. Не удалось. См. логи ниже                                                                                                                                                                                                                                                                                                        |                                                    |                           |                            |                           |                   |                     |                   |                  |                    |                      |                                        |                     |                                  |
| batch_size16_<br>gradacc16_<br>paddingfree_<br>flashatt_<br>ligerkernel     | Попробуем ligerkernel (библиотека готовых, вручную написанных  сверхбыстрых кернелов для конкретных операций). Не удалось. См. логи ниже                                                                                                                                                                                                                              |                                                    |                           |                            |                           |                   |                     |                   |                  |                    |                      |                                        |                     |                                  |
| batch_size16_<br>gradacc16_<br>seqpack                                      | Попробуем альтернативу padding free - sequence packing. Добавляет вместо паддингов (стакает) тексты между собой так, чтобы максимально эффективно их объединить, и на паддинг бы осталось минимально места. Должно ускорить относительно бейзлайна. Действительно ускорило по token per second, но по памяти больше. И padding free - оптимальнее                     | 10919.7683                                         |                           | 25.6117                    | 39.7324                   | 70.3627           | 6                   | 768344            | 10               | 0                  | 0                    |                                        | 11.2237             | 50694                            |

### **Возникшие проблемы:**

1) batch_size16_gradacc16_paddingfree_flashat_actoffload

```
   RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

2) batch_size16_gradacc16_paddingfree_flashatt_compile

```
   torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Triton Error [CUDA]: device kernel image is invalid

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information

You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
```

3) batch_size16_gradacc16_paddingfree_flashatt_compileinductor

```
   torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Triton Error [CUDA]: device kernel image is invalid

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information

You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
```

4) batch_size16_gradacc16_paddingfree_flashatt_ligerkernel

```
     File "/opt/conda/lib/python3.11/site-packages/triton/runtime/jit.py", line 653, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
    ^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/triton/compiler/compiler.py", line 395, in __getattribute__
    self._init_handles()
  File "/opt/conda/lib/python3.11/site-packages/triton/compiler/compiler.py", line 390, in _init_handles
    self.module, self.function, self.n_regs, self.n_spills = driver.active.utils.load_binary(
                                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Triton Error [CUDA]: device kernel image is invalid
```

### **Возможные причины проблем:**

Причины сбоев не установлены. Для torch.compile и Liger наблюдалась одинаковая ошибка загрузки Triton CUDA kernel, что позволяет предположить общую проблему совместимости окружения. Ошибка activation offloading также не позволяет оценить эффективность самого метода

В рамках ограниченного времени проблемы не были починены.

### **Полные запуски:**

В качестве лучших запусков были отобраны batch size 16, gradient accumulation 16, padding free и два варианта внимания - SDPA, Flash Attetnion. Соотношение tokens_per_second / tokens_per_second baseline в таблице. Оба варианта оптимизированы не только по времени, но и по потребляемой памяти

| run                                                                 | **tokens_<br>per_<br>second / tokens_<br>per_<br>second baseline** | train_<br>seconds | optimizer_<br>steps | target_<br>tokens | tokens_<br>per_<br>second_<br>including_<br>warmup | warmup_<br>steps | measured_<br>steps | measured_<br>seconds | tokens_<br>per_<br>second | seconds_<br>per_<br>optimizer_<br>step | peak_<br>allocated_<br>gib | peak_<br>reserved_<br>gib | validation_<br>loss | validation_<br>target_<br>tokens |
| ------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------- | ------------------- | ----------------- | -------------------------------------------------- | ---------------- | ------------------ | -------------------- | ------------------------- | -------------------------------------- | -------------------------- | ------------------------- | ------------------- | -------------------------------- |
| baseline                                                            | -                                                                  | 300.8527          | 370                 | 2106863           | 7002.9708                                          | 10               | 360                | 290.5995             | 7052.2466                 | 0.8072                                 | 41.2370                    | 43.4395                   | 6.5938              | 50694                            |
| batch_size16_<br>gradacc16_<br>paddingfree_<br>flashat_<br>fulldata | 2.5                                                                | 300.7947          | 231                 | 5289108           | 17583.7815                                         | 10               | 221                | 287.1811             | 17650.2644                | 1.2995                                 | 12.3008                    | 14.5703                   | 6.5351              | 50694                            |
| batch_size16_<br>gradacc16_<br>paddingfree_<br>sdpa_<br>fulldata    | 2.27                                                               | 300.9732          | 210                 | 4804797           | 15964.2035                                         | 10               | 200                | 286.3023             | 16012.8347                | 1.4315                                 | 12.5553                    | 14.9238                   | 6.7083              | 50694                            |
(fulldata означает, что полный запуск на 300 секунд)

* Loss: лучший финальный вариант - 6,5351 против 6,5938 у бейзлайна (вариант SDPA 6,7083)
* За одинаковые 300 секунд лучший вариант обработал примерно в 2,51 раза больше полезных токенов, но сделал меньше обновлений оптимизатора - 231 против 370
* Пиковая allocated-память снизилась с 41,24 до 12,30 GiB, то есть на 70,2%

### Комментарии по метрикам и методам

**Warm up и измерение скорости**

Для измерения скорости первые 10 шагов оптимизатора используются как warm up и не входят в расчёт tokens_per_second. Это позволяет уменьшить влияние начальных накладных расходов, например инициализации CUDA и первых запусков вычислительных ядер
Метрика tokens_per_second показывает количество полезных целевых токенов, обработанных за секунду после прогрева. Учитываются только токены, участвующие в расчёте loss (padding и другие позиции с игнорируемой меткой -100 исключаются). Скорость рассчитывается как число таких токенов за измеряемую часть обучения, делённое на её длительность. Метрика tokens_per_second_including_warmup дополнительно учитывает первые шаги и характеризует среднюю скорость за весь запуск

**Activation checkpointing**

Activation checkpointing уменьшает расход GPU памяти за счёт повторных вычислений. При обычном обучении модель сохраняет промежуточные активации после forward, чтобы использовать их для вычисления градиентов во время backward. При checkpointing часть этих активаций не сохраняется - когда они нужны для backward, соответствующий участок forward выполняется повторно. Поэтому метод экономит память, но может замедлять обучение

**SDPA, FlashAttention**

SDPA автоматически выбирает доступную реализацию attention и уже может использовать FlashAttention. Поэтому наблюдаемое ускорение нельзя объяснить только тем, что FlashAttention был включён явно. В двух конфигурациях могли отличаться конкретная реализация attention, работа с масками и обработка последовательностей разной длины.
В финальных запусках конфигурация с явно выбранным FlashAttention оказалась на 10,2% быстрее конфигурации SDPA