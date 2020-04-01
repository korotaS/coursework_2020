import json
import os


def parse(from_path, to_path):
    with open(from_path, 'r') as read:
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
    block_names = [name for name in block_names if 'block-' in name]
    for key in block_names:
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
    full_rl_data['agent'] = agent
    start_cond = planner_data['global-start']['conditions']
    goal_cond = planner_data['global-finish']['conditions']
    conditions = {'start': {block: [] for block in block_names},
                  'goal': {block: [] for block in block_names}}
    conditions['start'] = rewrite_conditions(start_cond, conditions['start'])
    conditions['goal'] = rewrite_conditions(goal_cond, conditions['goal'])
    full_rl_data['blocks'] = change_order_via_conditions(blocks, conditions)
    name = from_path.split('/')[-1]
    with open(to_path + 'parsed_' + name, 'w+') as write:
        write.write(json.dumps(full_rl_data, indent=4))


def rewrite_conditions(dict_from, dict_to):
    for key, value in dict_from.items():
        if 'onground' in key:
            dict_to[value['cause'][0]].append('onground')
        elif 'clear' in key:
            dict_to[value['cause'][0]].append('clear')
        elif 'on' in key:
            dict_to[value['cause'][0]].append({'on': value['cause'][1]})
    return dict_to


def contains_on(block_cond):
    for cond in block_cond:
        if type(cond) is dict and 'on' in cond:
            return cond['on']
    return None


def not_clear(conditions, block_name):
    for block, value in conditions.items():
        for cond in value:
            if type(cond) is dict and 'on' in cond and cond['on'] == block_name:
                return block
    return None


def change_order_via_conditions(blocks, conditions):
    block_names = list(blocks.keys())
    blocks_queue = []
    for name in block_names:
        if conditions['start'][name] == ['clear', 'onground'] and conditions['goal'][name] == ['clear', 'onground']:
            blocks_queue.append(name)
    for name in block_names:
        if name not in blocks_queue and contains_on(conditions['goal'][name]):
            base_block = contains_on(conditions['goal'][name])
            while contains_on(conditions['goal'][base_block]):
                base_block = contains_on(conditions['goal'][base_block])
            blocks_queue.append(base_block)
            top_block = base_block
            while not_clear(conditions['goal'], top_block):
                top_block = not_clear(conditions['goal'], top_block)
                blocks_queue.append(top_block)
    return {name: blocks[name] for name in blocks_queue}


dir_name = 'parsing_jsons/to_parse/'
to_path = 'parsing_jsons/parsed/'
for filename in os.listdir(dir_name):
    parse(dir_name + filename, to_path)
