import json
import os


def parse(path):
    with open(path, 'r') as read:
        planner_data = json.load(read)
    full_rl_data = {'map': {'walls': None}}
    map_size = planner_data['map']['map-size']
    full_rl_data['map']['rows'] = map_size[0]
    full_rl_data['map']['cols'] = map_size[1]
    start = planner_data['global-start']['objects']
    goal = planner_data['global-finish']['objects']
    agent = {'start_x': start['ag1']['x'],
             'start_y': start['ag1']['y'],
             'goal_x': goal['ag1']['x'],
             'goal_y': goal['ag1']['y'],
             'holding': planner_data['global-start']['ag1']['holding']['cause'][1]
             if 'holding' in planner_data['global-start']['ag1'] else None,
             'r': start['ag1']['r']}
    blocks = {}
    block_names = list(start.keys()) + ([agent['holding']] if agent['holding'] is not None else [])
    for key in block_names:
        if 'block-' not in key:
            continue
        blocks[key] = {}
        if key == agent['holding']:  # task is to put down a block
            s_item = start['ag1']
            g_item = goal[key]
            s_item['r'] = g_item['r']
        elif 'holding' in planner_data['global-finish']['ag1'] and \
             key == planner_data['global-finish']['ag1']['holding']['cause'][1]:  # task is to pick up a block
            s_item = start[key]
            g_item = goal['ag1']
            g_item['r'] = s_item['r']
        else:
            s_item = start[key]
            g_item = goal[key]
        blocks[key] = {'start_x': s_item['x'],
                       'start_y': s_item['y'],
                       'goal_x': g_item['x'],
                       'goal_y': g_item['y'],
                       'r': s_item['r']}
    full_rl_data['blocks'] = blocks
    full_rl_data['agent'] = agent
    name = path.split('/')[-1]
    with open('parsed/parsed_' + name, 'w+') as write:
        write.write(json.dumps(full_rl_data, indent=4))


dir_name = 'to_parse/'
for filename in os.listdir(dir_name):
    parse(dir_name + filename)
