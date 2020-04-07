# Курсовая работа по иерархическуму планированию с дообучением
Оригинальный репозиторий rl среды и q-learning агента: [[q-learning-with-options]](https://github.com/s-mawjee/q-learning-with-options)

### UPD 25.03.20:
Добавлена начальная версия среды BlocksWorld (пока что без Options).
### UPD 7.04.20:
Добавлен полный пайплайн, описание ниже

### Описание пайплайна
Всего у данного пайплайна 3 основных шага:
1. Работа планировщика: ему задается одна из spatial задач (которые есть в mapspatial/benchmarks/spatial/blocks/), он ее выполняет ее и сохраняет все шаги отдельно в tasks_jsons/task{task_num}/planner_steps/
2. Далее запускается planner_parser.py, который парсит шаги планировщика и переделывает их в таски для rl агента, объединяет некоторые шаги, изменяемые координаты которых находятся в определенном окне, и потом сохраняет эти новые таски в tasks_jsons/task{task_num}/planner_steps_parsed/
3. После этого запускается rl агент, который выполняет отдельно каждый распаршенный шаг в окне, а потом сохраняет ту стратегию, которую он выучил, в txt файлики в папку tasks_jsons/task{task_num}/f'rl_agent_steps/

### Запуск
Клонировать репозиторий и запустить train.py (в методе main() устанавливаются параметры запуска в переменной parameters).

### Dependencies
- Numpy
- Matplotlib
- Gym
- mapcore from glebkiselev [[map-core]](https://github.com/glebkiselev/map-core)
- mapspatial from glebkiselev [[map-spatial]](https://github.com/glebkiselev/map-spatial)
