import itertools
import logging
import re
import sys
from copy import deepcopy, copy
from functools import reduce
from mapcore.swm.src.components.semnet import Sign


def search_cm(events_list, signs, base='image'):
    searched = {}
    for event in events_list:
        for conn in event.coincidences:
            if conn.out_sign in signs:
                searched.setdefault(conn.out_sign, []).append(conn.get_out_cm(base))
    for s in signs:
        if not s in searched:
            searched[s] = None
    return searched

def pm_parser(pm, agent, signs, base = 'meaning'):
    pm_events = [ev for ev in pm.cause]
    searched = search_cm(pm_events, [signs['orientation'], signs['holding']], base = base)
    conditions = []

    holding = searched[signs['holding']]
    direction = None
    if holding:
        holding = holding[0]
    orientation = searched[signs['orientation']][0]
    for sign in orientation.get_signs():
        if sign.name != agent and sign.name != 'I':
            direction = sign
            break
    for ev in pm_events:
        if len(ev.coincidences) == 1:
            for con in ev.coincidences:
                if con.out_sign.name != "I" and con.out_sign.name != "orientation" and \
                        con.out_sign.name != "employment" and con.out_sign.name != "holding":
                        conditions.append(ev)
        else:
            ev_names = {s.name for s in ev.get_signs()}
            restrict = {'location', 'contain','handempty'}
            if ev_names.isdisjoint(restrict):
                conditions.append(ev)

    return conditions, direction, holding


def locater(location_name, map_size, objects, walls):
    dislocations = {}
    place_map = {}
    itera = 0
    start_x = map_size[0]
    start_y = map_size[1]

    blocksize = (map_size[2] - map_size[0]) / 3, (map_size[3] - map_size[1]) / 3

    for i in range(3):
        for j in range(3):
            fy = j * blocksize[1] + blocksize[1] + start_y
            fx = i * blocksize[0] + blocksize[0] + start_x
            dislocations[location_name + str(itera)] = [i * blocksize[0] + start_x, j * blocksize[1] + start_y, fx, fy]
            itera += 1

    for lkey, lvalue in dislocations.items():
        for ckey, cvalue in objects.items():
            object_x = cvalue['x']
            object_y = cvalue['y']
            if lvalue[0] <= object_x <= lvalue[2] and lvalue[1] <= object_y <= lvalue[3]:
                place_map.setdefault(lkey, set()).add(ckey)
            else:
                place_map.setdefault(lkey, set()).add(0)
        for wall in walls:
            if wall[1] <= lvalue[1] <= wall[3] and wall[0] <= lvalue[0] <= wall[2]:
                place_map.setdefault(lkey, set()).add('wall')
            elif wall[1] <= lvalue[3] <= wall[3] and wall[0] <= lvalue[2] <= wall[2]:
                place_map.setdefault(lkey, set()).add('wall')
            elif wall[3] <= lvalue[3] <= wall[1] and wall[0] <= lvalue[2] <= wall[2]:
                place_map.setdefault(lkey, set()).add('wall')
            elif lvalue[0] <= wall[0] <= lvalue[2] and wall[1] <= lvalue[1] <= wall[3]:
                place_map.setdefault(lkey, set()).add('wall')
            elif lvalue[0] <= wall[2] <= lvalue[2] and wall[1] <= lvalue[3] <= wall[3]:
                place_map.setdefault(lkey, set()).add('wall')
            elif lvalue[1] <= wall[1] <= lvalue[3] and wall[0] <= lvalue[0] <= wall[2]:
                place_map.setdefault(lkey, set()).add('wall')
            elif lvalue[1] <= wall[3] <= lvalue[3] and wall[0] <= lvalue[2] <= wall[2]:
                place_map.setdefault(lkey, set()).add('wall')
    for cell_name, signif in place_map.items():
        if len(signif) > 1 and 0 in signif:
            signif.remove(0)
    # remove border elements in cell, where less elements. Because clarification is needed
    for cell_name, signif in place_map.items():
        for element in signif:
            if element != 0 and not 'wall' in element:
                for cell_name2, signif2 in place_map.items():
                    if element in signif2 and cell_name != cell_name2:
                        if signif2 > signif:
                            signif.remove(element)
                            if len(signif) == 0:
                                signif.add(0)
                        else:
                            signif2.remove(element)
                            if len(signif2) == 0:
                                signif2.add(0)

    return dislocations, place_map

def scale(slist):
    newl = []
    for s in slist:
        if s != 0 and s % 3 > 0:
            s -= s % 3
            newl.append(s)
        elif s < 0 and s % 3 != 0:
            s += s % 3
            newl.append(s)
        else:
            newl.append(s)
    return newl


def size_founder(reg_loc, obj_loc, ag, border, cl_lv=0):
    others = set()
    target = None
    others.add(ag)
    reg_loc = scale(reg_loc)
    dislocations, place_map = locater('proto-cell', reg_loc, obj_loc, border)
    for cell, filling in place_map.items():
        if ag in filling:
            others = filling - others
            target = cell
            break
    if others:
        cl_lv += 1
        proto = scale(dislocations[target])
        size, cl_lv = size_founder(proto, obj_loc, ag, border, cl_lv)
    else:
        size = scale(dislocations[target])
        if not size[3] - size[1] > obj_loc[ag]['r'] or not size[2] - size[0] > obj_loc[ag]['r']:
            raise Exception('Can not place object! Too little space!')
            # sys.exit(1)
    return size, cl_lv


def belonging(cell, reg):
    Rx = (cell[2] - cell[0]) // 2
    Ry = (cell[3] - cell[1]) // 2
    a = reg[0] <= cell[0] and reg[1] <= cell[1]
    b = reg[2] >= cell[2] and reg[3] >= cell[3]
    if a and b:
        return True
    elif cell[0] <= reg[0] and cell[1] >= reg[1] and b:
        if reg[0] - cell[0] <= Rx:
            return True
    elif cell[0] >= reg[0] and cell[1] <= reg[1] and b:
        if reg[1] - cell[1] <= Ry:
            return True
    elif a and cell[2] >= reg[2] and cell[3] <= reg[3]:
        if cell[2] - reg[2] <= Rx:
            return True
    elif a and cell[2] <= reg[2] and cell[3] >= reg[3]:
        if cell[3] - reg[3] <= Ry:
            return True
    elif cell[0] <= reg[0] and cell[1] <= reg[1] and b:
        if reg[1] - cell[1] <= Ry and reg[0] - cell[0] <= Rx:
            return True
    elif cell[2] >= reg[2] and cell[3] >= reg[3] and a:
        if cell[2] - reg[2] <= Rx and cell[3] - reg[3] <= Ry:
            return True
    elif reg[0] < cell[0] and reg[1] > cell[1] and reg[2] < cell[2] and reg[3] > cell[3]:
        if cell[2] - reg[2] <= Rx and reg[1] - cell[1] <= Ry:
            return True
    elif reg[0] > cell[0] and reg[1] < cell[1] and reg[2] > cell[2] and reg[3] < cell[3]:
        if reg[0] - cell[0] <= Rx and cell[3] - reg[3] <= Ry:
            return True
    return False


def signify_connection(signs):
    Send = Sign("Send")
    send_signif = Send.add_significance()
    Broadcast = Sign("Broadcast")
    brdct_signif = Broadcast.add_significance()
    connector = brdct_signif.add_feature(send_signif)
    Send.add_out_significance(connector)
    Approve = Sign("Approve")
    approve_signif = Approve.add_significance()
    connector = approve_signif.add_feature(send_signif)
    Send.add_out_significance(connector)
    signs[Send.name] = Send
    signs[Broadcast.name] = Broadcast
    signs[Approve.name] = Approve

    They_sign = signs["They"]
    agents = They_sign.spread_up_activity_obj("significance", 1)
    agents_type = []
    for agent in agents:
        agents_type.append({cm.sign for cm in agent.sign.spread_up_activity_obj("significance", 1)})
    types = []
    if agents_type:
        types = [t for t in reduce(lambda x, y: x & y, agents_type) if t != signs["object"]]
    if types and len(agents):
        type = types[0]
    else:
        type = signs["I"]

    They_signif = They_sign.add_significance()
    brdct_signif = Broadcast.add_significance()
    connector = They_signif.add_feature(brdct_signif)
    Broadcast.add_out_significance(connector)
    type_signif = type.add_significance()
    approve_signif = Approve.add_significance()

    connector = type_signif.add_feature(approve_signif)
    Approve.add_out_significance(connector)

    brdct_signif = Broadcast.add_significance()
    executer = brdct_signif.add_feature(Broadcast.name.lower(), effect=True, actuator=True)
    Send.add_out_significance(executer)

    approve_signif = Approve.add_significance()
    executer = approve_signif.add_feature(Approve.name.lower(), effect=True, actuator=True)
    Send.add_out_significance(executer)


def adjoints(cdisl, rdisl):
    y_f = (cdisl[3] - cdisl[1]) // 2
    x_f = (cdisl[2] - cdisl[0]) // 2
    if rdisl[0] - x_f <= cdisl[2] <= rdisl[0] + x_f and rdisl[1] - y_f <= cdisl[3] <= rdisl[1] + y_f:
        return True
    elif rdisl[0] - x_f <= cdisl[0] and cdisl[2] <= rdisl[2] + x_f and rdisl[1] - y_f <= cdisl[3] <= rdisl[1] + y_f:
        return True
    elif rdisl[2] - x_f <= cdisl[0] <= rdisl[2] + x_f and rdisl[1] - y_f <= cdisl[3] <= rdisl[1] + y_f:
        return True
    elif rdisl[2] - x_f <= cdisl[0] <= rdisl[2] + x_f and rdisl[1] - y_f <= cdisl[1] and cdisl[3] <= rdisl[3] + y_f:
        return True
    elif rdisl[2] - x_f <= cdisl[0] <= rdisl[2] + x_f and rdisl[3] - y_f <= cdisl[1] <= rdisl[3] + y_f:
        return True
    elif rdisl[0] - x_f <= cdisl[0] and cdisl[2] <= rdisl[2] + x_f and rdisl[3] - y_f <= cdisl[1] <= rdisl[3] + y_f:
        return True
    elif rdisl[0] - x_f <= cdisl[2] <= rdisl[0] + x_f and rdisl[3] - y_f <= cdisl[1] <= rdisl[3] + y_f:
        return True
    elif rdisl[0] - x_f <= cdisl[2] <= rdisl[0] + x_f and rdisl[1] - y_f <= cdisl[1] and cdisl[3] <= rdisl[3] + y_f:
        return True
    return False


def cell_creater(size, obj_loc, region_location, wall, cl_lv=0):
    cell_loc = {}
    near_loc = {}
    ysize = size[3] - size[1]
    xsize = size[2] - size[0]
    new_region = [size[0] - xsize, size[1] - ysize, size[2] + xsize, size[3] + ysize]
    new_region = scale(new_region)
    cell_coords, cell_map = locater('cell-', new_region, obj_loc, wall)
    if len(cell_map['cell-4']) > 1:
        cl_lv += 1
        ysize = (size[3] - size[1]) // 3
        xsize = (size[2] - size[0]) // 3
        new_cell = [size[0] + xsize, size[1] + ysize, size[2] - xsize, size[3] - ysize]
        cell_loc, cell_map, near_loc, cell_coords, cl_lv = cell_creater(new_cell, obj_loc, region_location, wall, cl_lv)
    if not cell_loc and not near_loc:
        for cell, cdisl in cell_coords.items():
            for region, rdisl in region_location.items():
                if belonging(cdisl, rdisl):
                    cell_loc.setdefault(region, []).append(cell)
                    break
            else:
                cell_loc.setdefault('wall', []).append(cell)
            for region, rdisl in region_location.items():
                if adjoints(cdisl, rdisl):
                    near_loc.setdefault(cell, set()).add(region)
            if cell not in near_loc:
                near_loc.setdefault(cell, set()).add(0)

    return cell_loc, cell_map, near_loc, cell_coords, cl_lv


def signs_markup(parsed_map, static_map, agent, size=None, cl_lv=0):
    # place objects on map
    """
    :param parsed_map - dict vs objects
    :param agent - planning agent
    :param static_map - static map repr
    :return devision on regions, cells and cell_location
    """
    map_size = static_map.get('map-size')
    objects = parsed_map.get('objects')
    map_size = scale(map_size)
    # cl_lv = 0

    rmap = [0, 0]
    rmap.extend(map_size)

    # division into regions
    region_location, region_map = locater('region-', rmap, objects, static_map['wall'])

    # division into cells
    # cell size finding
    if not size:
        size = 0
        for key, value in region_map.items():
            if agent in value:
                new_val = deepcopy(value)
                new_val.remove(agent)
                if new_val:
                    try:
                        size, cl_lv = size_founder(region_location[key], objects, agent, static_map['wall'], cl_lv=1)
                    except Exception:
                        logging.info('Can not place object! Too little space!')
                        sys.exit(1)
                    break
                else:
                    size = region_location[key]
                    break
    else:
        if not size[0] <= objects[agent]['x'] <= size[2] or not size[1] <= objects[agent]['y'] <= size[3]:
            new_x = (size[2] - size[0]) / 2
            new_y = (size[3] - size[1]) / 2
            size = objects[agent]['x'] - new_x, objects[agent]['y'] - new_y, objects[agent]['x'] + new_x, \
                   objects[agent]['y'] + new_y
    cell_location, cell_map, near_loc, cell_coords, cl_lv = cell_creater(size, objects, region_location,
                                                                         static_map['wall'], cl_lv)

    return region_map, cell_map, cell_location, near_loc, cell_coords, size, cl_lv


def compare(s_matrix, pm, signs):
    if not pm.is_empty():
        iner_signs = pm.get_signs()
        causas = [el for el in s_matrix if 'cause' in el]
        effects = [el for el in s_matrix if 'effect' in el]
        elements = []
        if causas:
            for el in causas:
                elements.extend([obj[0] for obj in get_attributes(s_matrix[el], signs)])
        if effects:
            for el in effects:
                elements.extend([obj[0] for obj in get_attributes(s_matrix[el], signs)])
        if iner_signs == set(elements):
            return pm
    return None


def get_attributes(iner, signs):
    matrices = []
    if isinstance(iner, list):
        for s_name in iner:
            obj_sign = signs[s_name]
            obj_signif = obj_sign.significances[1]
            matrices.append((obj_sign, obj_signif))
        return matrices
    elif isinstance(iner, dict):
        for s_name, s_matrix in iner.items():
            if [el for el in iner if not 'cause' in el and not 'effect' in el]:
                if s_name not in signs:
                    s_name = [s for s in signs if s in s_name][0]
                el_sign = signs[s_name]
                pms = getattr(el_sign, 'significances')
                for index, pm in pms.items():
                    el_signif = compare(s_matrix, pm, signs)
                    if el_signif:
                        break
                else:
                    el_signif = el_sign.add_significance()
                    causas = [el for el in s_matrix if 'cause' in el]
                    effects = [el for el in s_matrix if 'effect' in el]
                    if causas:
                        elements = []
                        for el in causas:
                            elements.extend(get_attributes(s_matrix[el], signs))
                        for elem in elements:
                            connector = el_signif.add_feature(elem[1])
                            elem[0].add_out_significance(connector)
                    if effects:
                        elements = []
                        for el in effects:
                            elements.extend(get_attributes(s_matrix[el], signs))
                        for elem in elements:
                            connector = el_signif.add_feature(elem[1], effect=True)
                            elem[0].add_out_significance(connector)
                matrices.append((el_sign, el_signif))

        return matrices


def get_reg_location(cell_location, near_loc, region, signs):
    closely = [reg for reg in near_loc['cell-4'] if not reg == 0]
    nearly = set()
    for cell, regions in near_loc.items():
        if cell != 'cell-4':
            for reg in regions:
                nearly.add(reg)
    if region in cell_location:
        if 'cell-4' in cell_location[region]:
            return signs['include']
    if closely:
        if region in closely:
            return signs['closely']
    if region in nearly:
        return signs['nearly']
    return signs['faraway']


def resonated(signif, regions_struct, region, contain_reg, signs):
    inner_matrices = signif.spread_down_activity('significance', 5)
    if not inner_matrices:
        return False
    elif contain_reg == region:
        ssign = signif.get_signs()
        if signs['region?x'] in ssign and signs['cell?x'] in ssign:
            return True
    else:
        location = regions_struct[contain_reg][region][0]
        for matr in inner_matrices:
            if [s for s in matr if "region?y" in s.sign.name]:
                if matr[1].sign.name == location or matr[0].sign.name == location:
                    for m in inner_matrices:
                        if m[-1].sign.name == "cell-4":
                            return True
                break

    return False


def get_struct():
    regions = {}

    def itera(nclosely, nnearly):
        ncl = {}
        nnear = {}
        for nc in nclosely:
            ncl['region-' + str(nc[0])] = 'closely', nc[1]
        for nn in nnearly:
            nnear['region-' + str(nn[0])] = 'nearly', nn[1]
        ncl.update(nnear)
        return ncl

    ncl = itera([(1, 'below'), (3, 'right'), (4, 'below-right')],
                [(6, 'right'), (7, 'right'), (2, 'below'), (5, 'below'), (8, 'below-right')])
    regions.setdefault('region-0', {}).update(ncl)
    ncl = itera([(0, 'above'), (3, 'above-right'), (4, 'right'), (5, 'below-right'), (2, 'below')],
                [(6, 'right'), (7, 'right'), (8, 'right')])
    regions.setdefault('region-1', {}).update(ncl)
    ncl = itera([(1, 'above'), (4, 'above-right'), (5, 'right')],
                [(0, 'above'), (3, 'above'), (6, 'above-right'), (7, 'right'), (8, 'right')])
    regions.setdefault('region-2', {}).update(ncl)
    ncl = itera([(0, 'left'), (1, 'below-left'), (4, 'below'), (6, 'right'), (7, 'below-right')],
                [(2, 'below'), (5, 'below'), (8, 'below')])
    regions.setdefault('region-3', {}).update(ncl)
    ncl = itera([(0, 'above-left'), (1, 'left'), (2, 'below-left'), (3, 'above'), (5, 'below'), (6, 'above-right'),
                 (7, 'right'), (8, 'below-right')], [])
    regions.setdefault('region-4', {}).update(ncl)
    ncl = itera([(1, 'above-left'), (2, 'left'), (4, 'above'), (7, 'above-right'), (8, 'right')],
                [(0, 'above'), (3, 'above'), (6, 'above')])
    regions.setdefault('region-5', {}).update(ncl)
    ncl = itera([(3, 'left'), (4, 'below-left'), (7, 'below')],
                [(0, 'left'), (1, 'left'), (2, 'below-left'), (5, 'below'), (8, 'below')])
    regions.setdefault('region-6', {}).update(ncl)
    ncl = itera([(3, 'above-left'), (4, 'left'), (5, 'below-left'), (6, 'above'), (8, 'below')],
                [(0, 'left'), (1, 'left'), (2, 'left')])
    regions.setdefault('region-7', {}).update(ncl)
    ncl = itera([(4, 'above-left'), (7, 'above'), (5, 'left')],
                [(0, 'above-left'), (1, 'left'), (2, 'left'), (3, 'above'), (6, 'above')])
    regions.setdefault('region-8', {}).update(ncl)
    return regions


def state_prediction(agent, map, signs, holding=None, network = 'image'):
    agent_state = {}
    agent_state['name'] = agent
    if isinstance(map, dict):
        for predicate, value in map[agent.name].items():
            if predicate == 'orientation':
                agent_state['direction'] = signs[value]
            elif predicate == 'holding' or predicate == 'handempty':
                actuator = get_attributes({predicate:value}, signs)
                actuator.append('significance')
                agent_state['actuator'] = actuator
    elif isinstance(map, list):
        print('act - list')
        agent_state['direction'] = map[0]
        map[1].append('image')
        agent_state['actuator'] = map[1]
    else:
        agent_state['direction'] = map
        if holding:
            agent_state['actuator'] = [(holding.sign, holding), network]
        else:
            actuator = get_attributes(['handempty'], signs)
            actuator.append('significance')
            agent_state['actuator'] = actuator

    return agent_state


def define_map(map_name, region_map, cell_location, near_loc, regions_struct, signs):
    signs[map_name] = Sign(map_name)
    map_image = signs[map_name].add_image()
    elements = {}
    contain_sign = signs['contain']
    region_sign = signs['region']
    few_sign = signs['few']
    noth_sign = signs['nothing']
    location_signif = \
    [matr for _, matr in signs['location'].significances.items() if signs['direction'] in matr.get_signs()][0]
    contain_reg = [region for region, cells in cell_location.items() if 'cell-4' in cells][0]

    def get_or_add(sign):
        if sign not in elements:
            image = sign.add_image()
            elements[sign] = image
        return elements.get(sign)

    for region, objects in region_map.items():
        region_x = signs[region]
        flag = False
        if 0 in objects:
            flag = True
            objects.remove(0)
            objects.add("nothing")
            cont_signif = [signif for _, signif in contain_sign.significances.items() if
                           signs['region'] in signif.get_signs() and noth_sign in signif.get_signs()][0]
        else:
            cont_signif = [signif for _, signif in contain_sign.significances.items() if
                           signs['region'] in signif.get_signs() and signs['object'] in signif.get_signs()][0]
        connectors = []
        for object in objects:
            if object in signs:
                ob_sign = signs[object]
            else:
                ob_sign = Sign(object)
                signs[object] = ob_sign
                for s_name, sign in signs.items():
                    if s_name in object and s_name != object:
                        obj_signif = ob_sign.add_significance()
                        tp_signif = sign.add_significance()
                        connector = tp_signif.add_feature(obj_signif, zero_out=True)
                        ob_sign.add_out_significance(connector)
                        break

            ob_image = get_or_add(ob_sign)
            pm = cont_signif.copy('significance', 'image')

            region_image = region_x.add_image()
            pm.replace('image', region_sign, region_image)
            pm.replace('image', signs['object'], ob_image)
            if not flag:
                few_image = few_sign.add_image()
                pm.replace('image', signs['amount'], few_image)

            if connectors:
                con = connectors[0]
                connector = map_image.add_feature(pm, con.in_order)
            else:
                connector = map_image.add_feature(pm)
            contain_sign.add_out_image(connector)
            if not connectors:
                connectors.append(connector)

        loc_sign = get_reg_location(cell_location, near_loc, region, signs)
        connector = connectors[0]
        am = None
        for id, signif in loc_sign.significances.items():
            if resonated(signif, regions_struct, region, contain_reg, signs):
                am = signif.copy('significance', 'image')
                break
        if not am:
            print('Did not find applicable map')
            sys.exit(1)
        cell_image = signs["cell-4"].add_image()
        am.replace('image', signs["cell?x"], cell_image)
        inner_matrices = am.spread_down_activity('image', 3)
        for lmatrice in inner_matrices:
            if lmatrice[-1].sign.name == "region?z":
                reg_image = signs[region].add_image()
                am.replace('image', signs["region?z"], reg_image)
                break
        else:
            for lmatrice in inner_matrices:
                if lmatrice[-1].sign.name == "region?y":
                    reg_image = signs[region].add_image()
                    am.replace('image', signs["region?y"], reg_image)
        reg_image = signs[contain_reg].add_image()
        am.replace('image', signs["region?x"], reg_image)

        # direction matrice
        if contain_reg != region:
            dir_sign = signs[regions_struct[contain_reg][region][1]]
        else:
            dir_sign = signs["inside"]
        dir_matr = dir_sign.add_image()

        # location matrice
        location_am = location_signif.copy('significance', 'image')
        location_am.replace('image', signs['distance'], am)
        location_am.replace('image', signs['direction'], dir_matr)
        con = map_image.add_feature(location_am, connector.in_order)
        loc_sign.add_out_image(con)

    return map_image


def update_situation(sit_image, cell_map, signs, fast_est=None):
    # add ontable by logic
    if not fast_est:
        for cell, items in cell_map.items():
            if len(items) > 1:
                attributes = get_attributes(list(items), signs)
                roles = set()
                chains = {}
                for atr in attributes:
                    roles |= set(atr[0].find_attribute())
                    chains.setdefault(atr[0], set()).update(atr[0].spread_up_activity_obj('significance', 2))
                if len(roles) == len(attributes) or len(attributes) > 2:
                    predicates = {}
                    for role in roles:
                        predicates[role] = role.get_predicates()
                    com_preds = set()
                    for key1, item1 in predicates.items():
                        com_preds |= {s.name for s in item1}
                        for key2, item2 in predicates.items():
                            if key1 != key2:
                                for com in copy(com_preds):
                                    if not com in {s.name for s in item2}:
                                        com_preds.remove(com)

                    upper_roles = set()
                    for _, chain in chains.items():
                        upper_roles |= {ch.sign for ch in chain}
                    signifs = set()
                    for pred in com_preds:
                        for _, sig in getattr(signs[pred], "significances").items():
                            if len(sig.cause) == len(roles):
                                if sig.get_signs() <= upper_roles:
                                    signifs.add(sig)
                    if signifs:
                        signif = signifs.pop()
                        signif_signs = signif.get_signs()
                        replace = []
                        for sign in signif_signs:
                            for key, chain in chains.items():
                                if sign in [s.sign for s in chain]:
                                    replace.append((sign, key.add_image()))
                        merge_roles = []
                        for el1 in replace:
                            for el2 in replace:
                                if el1 != el2 and el1[0] != el2[0]:
                                    if el1[0] in signif_signs and el2[0] in signif_signs:
                                        if not (el1, el2) in merge_roles and not (el2, el1) in merge_roles:
                                            merge_roles.append((el1, el2))
                        matrices = []
                        for elem in merge_roles:
                            image = signif.copy('significance', 'image')
                            for pair in elem:
                                image.replace('image', pair[0], pair[1])
                            matrices.append(image)
                        for matrixe in matrices:
                            connector = sit_image.add_feature(matrixe)
                            matrixe.sign.add_out_image(connector)
    else:
        target_signs = ['holding', 'handempty', 'ontable']
        copied = {}
        for event in fast_est:
            if len(event.coincidences) == 1:
                for fevent in sit_image.cause:
                    if event.resonate('image', fevent):
                        break
                else:
                    if [con.get_out_cm('image').sign.name for con in event.coincidences][0] in target_signs:
                        sit_image.cause.append(event.copy(sit_image, 'image', 'image', copied))
                        # sit_meaning.add_event(event, False)
    return sit_image


def define_situation(sit_name, cell_map, conditions, agent_state, signs):
    signs[sit_name] = Sign(sit_name)
    sit_image = signs[sit_name].add_image()
    contain_sign = signs['contain']
    cellx = signs['cell?x']
    celly = signs['cell?y']
    few_sign = signs['few']

    location_signif = \
        [matr for _, matr in signs['location'].significances.items() if signs['direction'] in matr.get_signs()][0]
    cell_distr = {'cell-0': 'above-left', 'cell-1': 'left', 'cell-2': 'below-left', 'cell-3': 'above', \
                  'cell-5': 'below', 'cell-6': 'above-right', 'cell-7': 'right', 'cell-8': 'below-right'}

    mapper = {cell: value for cell, value in cell_map.items() if cell != 'cell-4'}

    agent = agent_state['name']
    orientation = agent_state['direction']
    actuator = agent_state['actuator']

    for cell, objects in mapper.items():
        noth_sign = signs['nothing']
        flag = False
        if 0 in objects:
            flag = True
            cont_signif = [signif for _, signif in contain_sign.significances.items() if
                           celly in signif.get_signs() and noth_sign in signif.get_signs()][0]
        else:
            cont_signif = [signif for _, signif in contain_sign.significances.items() if
                           celly in signif.get_signs() and signs['object'] in signif.get_signs()][0]
        cont_am = []
        connectors = []
        if flag:
            noth_image = noth_sign.add_image()
            cont_image = cont_signif.copy('significance', 'image')
            cont_image.replace('image', noth_sign, noth_image)
            cell_image = signs[cell].add_image()
            cont_image.replace('image', celly, cell_image)
            cont_am.append(cont_image)

        else:
            for obj in objects:
                obj_image = signs[obj].add_image()
                cont_image = cont_signif.copy('significance', 'image')
                cont_image.replace('image', signs['object'], obj_image)
                few_image = few_sign.add_image()
                cell_image = signs[cell].add_image()
                cont_image.replace('image', signs['amount'], few_image)
                cont_image.replace('image', celly, cell_image)
                cont_am.append(cont_image)

        for image in cont_am:
            if not connectors:
                connector = sit_image.add_feature(image)
                contain_sign.add_out_image(connector)
                connectors.append(connector)
            else:
                connector = sit_image.add_feature(image, connectors[0].in_order)
                contain_sign.add_out_image(connector)

        dir_y = signs[cell_distr[cell]]
        diry_image = dir_y.add_image()

        closely_signif = [signif for _, signif in signs['closely'].significances.items() if
                          cellx in signif.get_signs() and celly in signif.get_signs()][0]
        closely_image = closely_signif.copy('significance', 'image')
        cellx_image = signs['cell-4'].add_image()
        celly_image = signs[cell].add_image()
        closely_image.replace('image', cellx, cellx_image)
        closely_image.replace('image', celly, celly_image)

        location_image = location_signif.copy('significance', 'image')
        location_image.replace('image', signs['distance'], closely_image)
        location_image.replace('image', signs['direction'], diry_image)
        connector = sit_image.add_feature(location_image, connectors[0].in_order)
        location_image.sign.add_out_image(connector)

    empl_signif = [signif for _, signif in signs['employment'].significances.items() if cellx in signif.get_signs()][0]
    empl_image = empl_signif.copy('significance', 'image')
    cellx_image = signs['cell-4'].add_image()
    empl_image.replace('image', cellx, cellx_image)
    Ag_image = agent.add_image()
    empl_image.replace('image', signs['agent?ag'], Ag_image)
    conn = sit_image.add_feature(empl_image)
    empl_image.sign.add_out_image(conn)

    dir = signs['direction']
    orientation_image = orientation.add_image()
    orient_signif = [signif for _, signif in signs['orientation'].significances.items() if dir in signif.get_signs()][0]
    orient_image = orient_signif.copy('significance', 'image')
    orient_image.replace('image', dir, orientation_image)
    Ag_image = agent.add_image()
    orient_image.replace('image', signs['agent?ag'], Ag_image)
    conn = sit_image.add_feature(orient_image)
    orient_image.sign.add_out_image(conn)

    if actuator:
        act_img = actuator[0][1].copy(actuator[1], 'image')
        connector = sit_image.add_feature(act_img)
        act_img.sign.add_out_image(connector)
        if actuator[0][0].name == 'handempty':
            Ag_image = agent.add_image()
            conn = sit_image.add_feature(Ag_image, connector.in_order)
            agent.add_out_image(conn)

    if isinstance(conditions, dict):
        # in the grounding
        for predicate, signature in conditions.items():
            to_change = []
            signat = []
            pred_sm = None
            if len(signature['cause'])>1:
                for el in signature['cause']:
                    cms = [cm.sign.name for cm in signs[el].spread_up_activity_slice('significance', 1, 2)]
                    to_change.append(cms)
                for signa in itertools.product(*to_change):
                    if signa[0] == signa[1]:
                        continue
                    descr = {"cause":signa, "effect":[]}
                    pred_sm = get_predicate(predicate, descr, signs, fixplace=True)
                    if pred_sm:
                        signat = signa
                        break
                pred_im = pred_sm.copy('significances', 'image')
                for role_sign_name in signat:
                    obj_name = signature['cause'][signat.index(role_sign_name)]
                    obj_image = signs[obj_name].add_image()
                    pred_im.replace('image', signs[role_sign_name], obj_image)
                connector = sit_image.add_feature(pred_im)
                pred_im.sign.add_out_image(connector)
            else:
                pr_name = re.sub(r'[^\w\s{P}]+|[\d]+', r'', predicate).strip()
                pred_sm = signs[pr_name].significances[1]
                pred_im = pred_sm.copy('significances', 'image')
                obj_im = signs[signature['cause'][0]].add_image()
                connector = sit_image.add_feature(pred_im)
                pred_im.sign.add_out_image(connector)
                conn = sit_image.add_feature(obj_im, connector.in_order)
                obj_im.sign.add_out_image(conn)
    elif isinstance(conditions, list):
        # in copying from 1 sit to new one. Used in clarification, abstr and others
        for condition in conditions:
            event = condition.copy(sit_image, 'image', 'image', {})
            sit_image.add_event(event)

    # for event in events:
    #     copied = {}
    #     sit_image.cause.append(event.copy(sit_image, 'meaning', 'image', copied))

    global_situation = signs['situation']
    global_cm = global_situation.add_image()
    connector = global_cm.add_feature(sit_image)
    sit_image.sign.add_out_image(connector)

    return sit_image


def _clear_model(exp_signs, signs, structure):
    for type, smaller in list(structure.items()).copy():
        if exp_signs.get(type):
            if not type in signs:
                signs[type] = exp_signs[type]
            for object in smaller.copy():
                if exp_signs.get(object):
                    if not object in signs:
                        signs[object] = exp_signs[object]
                    smaller.remove(object)
            if not smaller:
                structure.pop(type)
    return signs, structure


def _ground_predicates(predicates, signs):
    # ground predicates
    for predicate_name, signature in predicates.items():
        variations = [pr for pr in signature if predicate_name in pr]
        if not predicate_name in signs:
            pred_sign = Sign(predicate_name)
            signs[predicate_name] = pred_sign
        else:
            pred_sign = signs[predicate_name]
        if 'cause' and 'effect' in signature:
            pred_signif = pred_sign.add_significance()
            if len(signature['cause']) > 1:
                for part, iner in signature.items():
                    matrices = get_attributes(iner, signs)
                    for el in matrices:
                        if part == "cause":
                            connector = pred_signif.add_feature(el[1], effect=False, zero_out=True)
                        else:
                            connector = pred_signif.add_feature(el[1], effect=True, zero_out=True)
                        el[0].add_out_significance(connector)
        elif variations:
            for part, iner in signature.items():
                mixed = []
                matrices = []
                pred_signif = pred_sign.add_significance()
                for element in iner:
                    effect = True
                    if 'cause' in element: effect = False
                    if not 'any' in iner[element]:
                        matrices = get_attributes(iner[element], signs)
                        for el in matrices:
                            connector = pred_signif.add_feature(el[1], effect=effect, zero_out=True)
                            el[0].add_out_significance(connector)
                    else:
                        for key, el in iner[element].items():
                            matrices.extend(get_attributes(el, signs))
                        mixed.append(matrices)
                        matrices = []

                if mixed:
                    combinations = itertools.product(mixed[0], mixed[1])
                    history = []
                    for element in combinations:
                        el_signs = element[0][0], element[1][0]
                        if not element[0][0] == element[1][0] and el_signs not in history:
                            history.append(el_signs)
                            history.append((el_signs[1], el_signs[0]))
                            pred_signif = pred_sign.add_significance()
                            connector = pred_signif.add_feature(element[0][1], effect=False)
                            element[0][0].add_out_significance(connector)
                            connector = pred_signif.add_feature(element[1][1], effect=True)
                            element[1][0].add_out_significance(connector)

def get_predicate(pr_name, pr_model, signs, fixplace = False):
    pr_name = re.sub(r'[^\w\s{P}]+|[\d]+', r'', pr_name).strip()
    pr_sign = signs[pr_name]
    model_signs = [el for el in itertools.chain(pr_model['cause'], pr_model['effect'])]
    for id, matrice in pr_sign.significances.items():
        matrice_signs = [s.name for s in matrice.get_signs()]
        if len(matrice_signs) == len(model_signs):
            if not fixplace:
                for el in model_signs:
                    if not el in matrice_signs:
                        break
                else:
                    return matrice
            else:
                flag = False
                for el in model_signs:
                    for conn in matrice.cause[model_signs.index(el)].coincidences:
                        if el != conn.out_sign.name:
                            flag = True
                            break
                    if flag:
                        break
                else:
                    return matrice
    return None

def _ground_actions(actions, obj_means, constraints, signs, agents, agent_type):
    events = []
    # ground actions
    for action_name, smaller in actions.items():
        if not action_name in signs:
            act_sign = Sign(action_name)
            signs[action_name] = act_sign
            act_signif = act_sign.add_significance()
        else:
            act_sign = signs[action_name]
            act_signif = act_sign.add_significance()

        for part, decription in smaller.items():
            effect = True
            if part == "cause": effect = False
            for predicate, signature in decription.items():
                if len(signature['cause'])>1:
                    pred = get_predicate(predicate, signature, signs)
                    if pred:
                        connector = act_signif.add_feature(pred, effect=effect)
                        pred.sign.add_out_significance(connector)
                    else:
                        raise Exception(
                            "predicate *{0}* has wrong definition in action *{1}*".format(predicate, action_name))
                elif len(signature['cause']) == 1:
                    pred = signs[predicate].significances[1]
                    connector = act_signif.add_feature(pred, effect=effect)
                    pred.sign.add_out_significance(connector)
                    role_sign = signs[signature["cause"][0]]
                    conn = act_signif.add_feature(role_sign.significances[1], connector.in_order, effect=effect)
                    role_sign.add_out_significance(conn)

        if not constraints:
            for ag in agents:
                act_mean = act_signif.copy('significance', 'meaning')
                ag_sign = signs[ag]
                act_mean.replace('meaning', signs[agent_type], obj_means[ag_sign])
                connector = act_mean.add_feature(obj_means[ag_sign])
                efconnector = act_mean.add_feature(obj_means[ag_sign], effect=True)
                events.append(act_mean.effect[abs(efconnector.in_order) - 1])
                ag_sign.add_out_meaning(connector)
        else:
            events.extend(constr_replace(constraints, act_signif, agents))
    return events

def constr_replace(constraints, act_signif, agents):
    act_mean = act_signif.copy('significance', 'meaning')
    pass

def state_fixation(state, el_coords, signs, element):
    """
    :param state: cm of described state on meanings network
    :param el_coords: coords from signs_markup function
    :param signs: active sign base
    :return: state matrix on images network
    """
    im = state.sign.add_image()

    for number in range(9):
        cs = signs[element + '-' + str(number)]
        cimage = cs.add_image()
        coords = el_coords[element + '-' + str(number)]
        cimage.add_feature(coords, effect=False, view=True)
        connector = im.add_feature(cimage, effect=False)
        cs.add_out_image(connector)

    return im

def tree_refinement(line, opendelim, closedelim):
    stack = []
    for m in re.finditer(r'[{}{}]'.format(opendelim, closedelim), line):
        pos = m.start()
        CL = line[pos:pos+len(opendelim)]
        ABS = line[pos: pos+len(closedelim)]
        if CL == opendelim:
            iter = pos + len(opendelim)+1
            curs = line[iter]
            num = ''
            while curs != ' ':
                num+=curs
                if iter+1 == len(line)-1:
                    break
                else:
                    iter+=1
                    curs = line[iter]
            stack.append(eval(num))

        elif ABS == closedelim:
            iter = pos + len(closedelim)+1
            curs = line[iter]
            num = ''
            while curs != ' ' or curs != "'":
                num+=curs
                if iter+1 == len(line)-1:
                    break
                else:
                    iter+=1
                    curs = line[iter]
            #stack.append(eval(num))
            if len(stack) > 0:
                prevpos = stack.pop()
                yield prevpos, eval(num), len(stack)
            else:
                # error
                print("encountered extraneous closing quote at pos {}: '{}'".format(pos, line[pos:]))
                pass

    if len(stack) > 0:
        for pos in stack:
            print("expecting closing quote to match open quote starting at: '{}'"
                  .format(line[pos - 1:]))