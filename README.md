# Курсовая работа по иерархическуму планированию с дообучением

### Описание пайплайна
Всего у данного пайплайна такие основные шаги:
1. Работа планировщика: ему задается одна из spatial задач (которые есть в mapspatial/benchmarks/spatial/blocks/), он ее выполняет ее и сохраняет все шаги отдельно в tasks_jsons/task{task_num}/planner_steps/
2. Далее запускается utils/planner_parser.py, который парсит шаги планировщика и переделывает их в таски для rl агента, объединяет некоторые шаги, изменяемые координаты которых находятся в определенном окне, и потом сохраняет эти новые таски в tasks_jsons/task{task_num}/planner_steps_parsed/
3. После этого запускается rl агент, который выполняет отдельно каждый распаршенный шаг в окне, а потом сохраняет ту стратегию, которую он выучил, в txt файлики в папку tasks_jsons/task{task_num}/'rl_agent_steps/
4. Затем из шагов rl агента достаются те шаги, которые содержат действия pick-up или put down, и сохраняются в tasks_jsons/task{task_num}/manipulator_sits_raw/situations.json
5. На последнем шаге запускается DQN-агент, который находит наилучшие последовательности движений манипулятора и сохраняет их в tasks_jsons/task{task_num}/manipulator_sits_solved/

### Запуск
Клонировать репозиторий и запустить train.py (в методе main() устанавливаются параметры запуска в переменной parameters).

### Dependencies
- Numpy
- Matplotlib
- Gym
- mapcore from glebkiselev [[map-core]](https://github.com/glebkiselev/map-core)
- mapspatial from glebkiselev [[map-spatial]](https://github.com/glebkiselev/map-spatial)
