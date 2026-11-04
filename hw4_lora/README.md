# Дообучение Qwen на GSM8K

## Данные и протокол

Используется GSM8K (`openai/gsm8k`, `main`): **7473 задачи train и 1319 задач test**, с условиями, пошаговыми решениями и числовыми ответами после `####`. Размеры и примеры — в [dataset.json][dataset]. Test не участвует в обучении и выборе чекпойнта.

Модель — **Qwen2.5-1.5B-Instruct**. Вход оформляется через chat template, loss считается только по ответу ассистента. Во всех обученных вариантах: 3 эпохи, seed 42, batch size 16 на устройство, gradient accumulation 4, bf16, max length 2048. Learning rate: `2e-5` для [Full SFT](full_sft.py), `2e-4` для [LoRA](lora.py). У LoRA dropout 0,05; по умолчанию адаптеры стоят в `q_proj` и `v_proj`, в варианте «все проекции» — также в `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. 

## Результаты

Оценка выполнена на всех 1319 задачах test, генерация жадная, максимум 512 новых токенов. **Accuracy** — доля верных чисел; **parse rate** — доля распознанных ответов; **MAE parsed** — их средняя абсолютная ошибка. **Perplexity** — по эталонным решениям при teacher forcing. Реализация оценки: [Full SFT](full_sft.py#L135), [LoRA](lora.py#L187).

| Модель / конфигурация | Accuracy, % ↑ | Parse rate, % ↑ | MAE parsed ↓ | Perplexity ↓ | Источники | Вывод |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| [Исходная Qwen](full_sft.py) | 1.29 | 1.97 | 1,206.52 | 1.8354 | [метрики][base-metrics], [конфиг][base-config] | Распознаны лишь 26 ответов; accuracy ограничена форматом. |
| [Full SFT](full_sft.py) | 49.51 | 99.39 | 2,856.04 | 1.5432 | [метрики][sft-metrics], [конфиг][sft-config] | 653 верных ответа; минимальная MAE среди обученных вариантов. |
| [LoRA r=16, α=32](lora.py) | 49.81 | 99.17 | 3,926.44 | 1.5514 | [метрики][default-metrics], [конфиг][default-config] | 657 верных ответов — максимум в этой серии; на 4 больше Full SFT. |
| [LoRA r=8, α=32](lora.py) | 48.29 | 98.86 | 4,603.51 | 1.5520 | [метрики][r8-metrics], [конфиг][r8-config] | Параметров вдвое меньше базовой LoRA, верных ответов — на 20. |
| [LoRA все проекции, r=16, α=32](lora.py) | 48.60 | 99.47 | 79,495.18 | 1.5421 | [метрики][all-metrics], [конфиг][all-config] | Минимальная perplexity, но accuracy ниже базовой LoRA; затраты выше. |
| [LoRA r=2, α=32](lora.py) | 49.28 | 99.01 | 81,491.05 | 1.5533 | [метрики][r2-metrics], [конфиг][r2-config] | Параметров в 8 раз меньше, верных ответов — на 7; MAE выше. |
| [LoRA r=16, α=8](lora.py) | 49.73 | 97.80 | 3,093.80 | 1.5583 | [метрики][alpha8-metrics], [конфиг][alpha8-config] | На 1 верный ответ меньше базовой LoRA, но нераспознанных ответов больше: 29 вместо 11. |

`mae_all = null` во всех строках из-за нераспознанных ответов. Парсер принимает число после `####` или отдельное число. У baseline 1293 ответа не распознаны: прирост accuracy отражает и освоение формата, а не только рассуждений. MAE сравнивается с учётом parse rate и масштаба чисел. Для проверки: [ответы baseline][base-predictions] и [ответы базовой LoRA][default-predictions] для проверки.

### Вычислительные затраты

Время — только `trainer.train()`. Память — пик **allocated / reserved** PyTorch на текущем CUDA-устройстве (GiB), без суммирования GPU. [Окружение][environment]: A100-SXM4-80GB, PyTorch 2.8.0+cu126. Ссылки ведут в `resources.json`.

| Вариант / источник | Обучаемые параметры | Доля, % | Время, с | Пик allocated / reserved, GiB |
| --- | ---: | ---: | ---: | ---: |
| [Full SFT][sft-resources] | 1,543,714,304 | 100.000 | 320.2 | 53.67 / 75.69 |
| [LoRA r=16, α=32][default-resources] | 2,179,072 | 0.141 | 291.1 | 41.39 / 76.41 |
| [LoRA r=8, α=32][r8-resources] | 1,089,536 | 0.071 | 289.9 | 41.36 / 76.39 |
| [LoRA все проекции, r=16, α=32][all-resources] | 18,464,768 | 1.182 | 499.0 | 59.85 / 78.05 |
| [LoRA r=2, α=32][r2-resources] | 272,384 | 0.018 | 289.9 | 41.33 / 76.38 |
| [LoRA r=16, α=8][alpha8-resources] | 2,179,072 | 0.141 | 289.9 | 41.39 / 76.41 |

Базовая LoRA дала близкое к Full SFT качество при **0,141% обучаемых параметров**, примерно на 9% меньшем времени и на 23% меньшем пике allocated-памяти. Reserved-память при этом почти не уменьшилась. Расширение набора проекций не улучшило accuracy и увеличило затраты. Снижение ранга не дало монотонного изменения качества: при фиксированном α меняется также масштаб α/r. Один seed и сравнение вариантов на одном test позволяют сделать только предварительные выводы.

## Артефакты и запуск

В [runs](runs) сохранены метрики, предсказания, конфиги и логи 7 запусков. Модели и кеш удалены и исключены через [.gitignore](.gitignore).

Статус обученных запусков — `failed` из-за отсутствующего `base_predictions` при формировании отчёта. Все 177 шагов и оценка завершены ([лог][sft-log]), но итоговые `results.json`, `comparison.json` и `manual_review.json` не созданы. Запуск с baseline позже упал при обучении из-за OOM.

Конфигурация задаётся в `CONFIG`: [Full SFT](full_sft.py#L22), [LoRA](lora.py#L26). Команды из этой папки:

```bash
python full_sft.py
python lora.py
```

В `lora.py` сейчас α=8; остальные настройки — в конфиге строки таблицы. Baseline отключён в обоих скриптах; описанная ошибка отчёта остаётся в коде.

[base-metrics]: runs/20261104_175754_974354_full_sft/metrics_base.json
[base-config]: runs/20261104_175754_974354_full_sft/config.json

[sft-metrics]: runs/20261104_181346_279891_full_sft/metrics_tuned.json
[sft-config]: runs/20261104_181346_279891_full_sft/config.json
[sft-resources]: runs/20261104_181346_279891_full_sft/resources.json

[default-metrics]: runs/20261104_182530_149359_lora_default/metrics_tuned.json
[default-config]: runs/20261104_182530_149359_lora_default/config.json
[default-resources]: runs/20261104_182530_149359_lora_default/resources.json

[r8-metrics]: runs/20261104_183522_854442_lora_r8/metrics_tuned.json
[r8-config]: runs/20261104_183522_854442_lora_r8/config.json
[r8-resources]: runs/20261104_183522_854442_lora_r8/resources.json

[all-metrics]: runs/20261104_184601_572192_lora_all_projections/metrics_tuned.json
[all-config]: runs/20261104_184601_572192_lora_all_projections/config.json
[all-resources]: runs/20261104_184601_572192_lora_all_projections/resources.json

[r2-metrics]: runs/20261104_190120_293326_lora_r2/metrics_tuned.json
[r2-config]: runs/20261104_190120_293326_lora_r2/config.json
[r2-resources]: runs/20261104_190120_293326_lora_r2/resources.json

[alpha8-metrics]: runs/20261104_191121_204841_lora_alpha8/metrics_tuned.json
[alpha8-config]: runs/20261104_191121_204841_lora_alpha8/config.json
[alpha8-resources]: runs/20261104_191121_204841_lora_alpha8/resources.json
[base-predictions]: runs/20261104_175754_974354_full_sft/predictions_base.json
[default-predictions]: runs/20261104_182530_149359_lora_default/predictions_tuned.json
[dataset]: runs/20261104_181346_279891_full_sft/dataset.json
[environment]: runs/20261104_181346_279891_full_sft/environment.json
[sft-log]: runs/20261104_181346_279891_full_sft/run.log
