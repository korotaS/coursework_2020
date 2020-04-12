import argparse
import os
import platform
import json

from .config_master import create_config, get_config
from mapspatial.mapplanner import MapPlanner


def main(args, task_num, solution_save_path):
    if platform.system() != 'Windows':
        delim = '/'
    else:
        delim = '\\'
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument(dest='problem', nargs='?')
    argparser.add_argument(dest='agpath', nargs='?')
    argparser.add_argument(dest='agtype', nargs='?')
    argparser.add_argument(dest='backward', nargs='?')
    argparser.add_argument(dest='config_path', nargs='?')
    args = argparser.parse_args(args)
    if args.problem and args.agpath and args.agtype:
        if not args.config_path:
            path = create_config(benchmark=os.path.abspath(args.problem), delim=delim,
                                 task_type='spatial', agpath = args.agpath, agtype = args.agtype, backward=args.backward)
        else:
            path = args.config_path
    else:
        if not args.config_path:
            path = create_config(task_num=task_num, delim=delim, backward='False', task_type='spatial')
        else:
            path = args.config_path

    # after 1 time creating config simply send a path
    planner = MapPlanner(**get_config(path))
    solution = planner.search()
    save_steps(solution, planner.problem, solution_save_path)
    return solution


def save_steps(solution, problem_path, save_path):
    with open(problem_path, 'r') as read:
        map_data = json.load(read)['map']
    situations = []
    for i in solution:
        for j in i.values():
            for k in j:
                if k[1] != 'Clarify':
                    situations.append({'map': map_data,
                                       'global-start': k[6][0],
                                       'global-finish': k[6][1]})
    for i, sit in enumerate(situations):
        with open(save_path + f'/{i}.json', 'w+') as write:
            write.write(json.dumps(sit, indent=4))


# if __name__ == '__main__':
#     main(sys.argv[1:], '0')
