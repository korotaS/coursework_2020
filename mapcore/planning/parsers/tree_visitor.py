#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

"""
Classes and functions that traverse the PDDL abstract syntax tree (AST)
generated by pddl_parser.py, thereby creating the PDDL data structure.
"""

from mapcore.planning.parsers import pddl


class SemanticError(Exception):
    """Exception indicating an error during traversal of AST."""

    def __init__(self, value):
        """Constructor of SemanticError.

        Keyword arguments:
        value -- the error message
        """
        self.value = value

    def __str__(self):
        return repr(self.value)


class Visitable:
    """
    The Visitable class is part of the Visitor Pattern. Every AST node created
    by the Parser derives from this class.

    The accept-method calls the appropriate method of the visitor.
    """
    def __init__(self, vname=None):
        """Constructor of the Visitable.

        Keyword arguments:
        vname -- the name of the of the callback that will be called on the
                 visitor instance
        """
        self._visitorName = vname

    def accept(self, visitor):
        if self._visitorName == None:
            raise ValueError('Error: visit method of uninitialized visitor '
                             'called!')
        # get the appropriate method of the visitor instance
        m = getattr(visitor, self._visitorName)
        # ensure that the method is callable
        if not hasattr(m, '__call__'):
            raise ValueError('Error: cannot call undefined method: %s on '
                             'visitor' % self._visitorName)
        # and finally call the callback
        m(self)


class PDDLVisitor:
    """
    The standard tree traversal PDDL Visitor from which other Visitors derive.

    In each node, the Visitor just applies itself to all its children.
    """
    def visit_domain_def(self, node):
        node.requirements.accept(self)
        if node.types != None:
            for t in node.types:
                t.accept(self)
        if node.constants != None:
            for c in node.constants:
                c.accept(self)
        node.predicates.accept(self)
        if node.actions != None:
            for a in node.actions:
                a.accept(self)

    def visit_problem_def(self, node):
        for o in node.objects:
            o.accept(self)
        node.init.accept(self)
        node.goal.accept(self)

    def visit_predicates_stmt(self, node):
        for p in node.predicates:
            p.accept(self)

    def visit_action_stmt(self, node):
        for p in node.parameters:
            p.accept(self)
        node.precond.accept(self)
        node.effect.accept(self)

    def visit_formula(self, node):
        for c in node.children:
            c.accept(self)

    def visit_type(self, node):
        return  # nothing to do right now

    def visit_effect_stmt(self, node):
        node.formula.accept(self)

    def visit_precondition_stmt(self, node):
        node.formula.accept(self)

    def visit_requirements_stmt(self, node):
        for k in node.keywords:
            k.accept(self)

    def visit_predicate(self, node):
        for p in node.parameters:
            p.accept(self)

    def visit_variable(self, node):
        pass

    def visit_init_stmt(self, node):
        for p in node.predicates:
            p.accept(self)

    def visit_goal_stmt(self, node):
        node.formula.accept(self)

    def visit_predicate_instance(self, node):
        return  # nothing to do right now

    def visit_object(self, node):
        return  # nothing to do right now

    def visit_keyword(self, node):
        return  # nothing to do right now


class TraversePDDLDomain(PDDLVisitor):
    """The PDDL-domain Visitor.

    Expands the functionality of the PDDLVisitor to traversal of ASTs that
    represent a PDDL-domain file. It results in the PDDL data structure
    (pddl.py) representation of the domain file.
    """

    def get_in(self, node):
        """
        Helper method to access a global hash in which information for each
        node in the AST can be stored.
        """
        return self._nodeHash[node]

    def set_in(self, node, val):
        """
        Helper method to write a global hash in which information for each node
        in the AST can be stored.
        """
        self._nodeHash[node] = val

    def __init__(self):
        self._types = dict()
        self._predicates = dict()
        self._nodeHash = dict()
        self._requirements = set()
        self._actions = dict()
        self.domain = None
        self._objectType = pddl.Type('object', None)
        self._constants = dict()

    def visit_domain_def(self, node):
        """Visits a PDDL domain definition."""
        explicitObjectDef = False

        # Requirements statement is optional.
        if node.requirements:
            node.requirements.accept(self)

        # Visit all type definitions.
        if node.types != None:
            for t in node.types:
                if t.name == 'object':
                    explicitObjectDef = True
                t.accept(self)
                type = self.get_in(t)
                self._types[type.name] = type
        # Add the default object type to the type definitions,
        # if it was not explicitly created.
        if not explicitObjectDef:
            self._types['object'] = self._objectType

        # Link all types to their parent types directly.
        for t in self._types.values():
            # Object type has no parent.
            if t.name == 'object':
                continue
            if not t.parent in self._types:
                raise SemanticError('Error unknown parent type: ' + t.parent)
            t.parent = self._types[t.parent]

        # Visit all predicates.
        node.predicates.accept(self)

        # Visit all actions.
        if node.actions != None:
            for a in node.actions:
                a.accept(self)
                action = self.get_in(a)
                if action.name in self._actions:
                    raise SemanticError('Error: action with name ' +
                                        action.name +
                                        ' has already been defined')
                self._actions[action.name] = action

        # Visit all constants.
        if node.constants != None:
            for c in node.constants:
                c.accept(self)

        # Finally generate PDDL domain data structure.
        self.domain = pddl.Domain(node.name, self._types, self._predicates,
                                  self._actions, self._constants)

    def visit_object(self, node):
        """Visits a PDDL object definition."""
        type_name = node.typeName
        if type_name == None:
            type_name = 'object'
        if not type_name in self._types:
            raise SemanticError('Error: unknown type ' + type_name +
                                ' used in object definition!')
        if node.name in self._constants:
            raise SemanticError('Error: multiple defines of object with '
                                'name ' + node.name)
        # Add constant with its corresponding type to the constants dict.
        self._constants[node.name] = self._types[type_name]

    def visit_type(self, node):
        """Visits a PDDL type definition."""
        # Store matching parent type in node
        # (if none is given, it's always object)
        if node.parent == None:
            self.set_in(node, pddl.Type(node.name, 'object'))
        else:
            self.set_in(node, pddl.Type(node.name, node.parent))

    def visit_requirements_stmt(self, node):
        """Visits a PDDL requirement statement."""
        # Visit all requirement keywords...
        for k in node.keywords:
            k.accept(self)
            requirementName = self.get_in(k)
            # ... and add them to the requirement list.
            self._requirements.add(requirementName)

    def visit_keyword(self, node):
        """Visits a PDDL keyword."""
        # Nothing to do but to store its name in the node.
        self.set_in(node, node.name)

    def visit_predicates_stmt(self, node):
        """Visits a PDDL predicate statement."""
        # Visit all predicates in the predicate statement.
        for p in node.predicates:
            p.accept(self)
            predicate = self.get_in(p)
            # Check for duplicate predicate definitions.
            if predicate.name in self._predicates:
                raise SemanticError('Error predicate with name ' +
                                    predicate.name +
                                    ' has already been defined')
            # Add to predicate list.
            self._predicates[predicate.name] = predicate

    def visit_predicate(self, node):
        """Visits a PDDL predicate."""
        signature = list()
        # Visit all predicate parameters.
        for v in node.parameters:
            v.accept(self)
            signatureTuple = self.get_in(v)
            # Append each parameter to the predicate signature.
            signature.append(signatureTuple)
        # Create new PDDL predicate and store it in node.
        self.set_in(node, pddl.Predicate(node.name, signature))

    def visit_variable(self, node):
        """Visits a PDDL variable."""
        # If there is no type given, its always of type 'object'.
        if not node.typed:
            self.set_in(node, (node.name, [self._types['object']]))
        else:
            # Visit all type declarations of the variable.
            typelist = list()
            for t in node.types:
                # Check whether they have been defined.
                if not t in self._types:
                    raise SemanticError('Error unknown type ' + t +
                                        ' used in predicate definition')
                typelist.append(self._types[t])
            # Store variable information (var_name, tuple(types)) in node.
            self.set_in(node, (node.name, tuple(typelist)))

    def visit_action_stmt(self, node):
        """Visits a PDDL action statement."""
        signature = list()
        # Visit all parameters and create signature.
        for v in node.parameters:
            v.accept(self)
            signatureTuple = self.get_in(v)
            signature.append(signatureTuple)

        # Visit the precondition statement.
        node.precond.accept(self)
        precond = self.get_in(node.precond)

        # Visit the effect statement.
        node.effect.accept(self)
        effect = self.get_in(node.effect)

        # Give agent type
        agents = []
        if node.agent:
            for agent in node.agent:
                agents.append(agent.types[0])

        # Create new PDDL action and store in node.
        self.set_in(node, pddl.Action(node.name, agents, signature, precond, effect))

    def add_precond(self, precond, c):
        """Helper function for visit_precondition_stmt.

        Keyword arguments:
        precond -- a list of preconditions
        c -- the formula representing a precondition we want to add to the list
        """
        from mapcore.planning.parsers.pddl_parser import Variable
        predDef = self._predicates[c.key]
        signature = list()
        count = 0
        # Check for correct number of arguments.
        if len(c.children) != len(predDef.signature):
            raise SemanticError('Error: wrong number of arguments for '
                                'predicate ' + c.key + ' in precondition of '
                                'action')
        # Apply to all arguments.
        for v in c.children:
            if isinstance(v.key, Variable):
                signature.append((v.key.name, predDef.signature[count][1]))
            else:
                signature.append((v.key, predDef.signature[count][1]))
            count += 1

        # Add predicate to precondition list.
        precond.append(pddl.Predicate(c.key, signature))

    def visit_precondition_stmt(self, node):
        """ Visits a PDDL precondition statement."""
        precond = list()
        formula = node.formula
        # For now we only allow and in the precondition.
        if formula.key == 'and':
            # Apply to all predicates in precondition.
            for c in formula.children:
                if not isinstance(c.key, str):
                    raise SemanticError('Error predicate with non str key: ' +
                                        ''.join([c2.key.name + ' '
                                                for c2 in formula.children]))
                # Check whether predicate was defined.
                if not c.key in self._predicates:
                    raise SemanticError('Error unknown predicate ' + c.key +
                                        ' used in precondition of action')
                # Call helper.
                self.add_precond(precond, c)
        else:
            # If not 'and' we only allow a single predicate in precondition.
            if not formula.key in self._predicates:
                raise SemanticError('Error: predicate in precondition is not '
                                    'in CNF')
            # Call helper.
            self.add_precond(precond, formula)
        self.set_in(node, precond)

    def add_effect(self, effect, c):
        """Helper function for visit_effect_stmt.

        Keyword arguments:
        effect -- instance of the effect data structure
        c -- the formula representing the effect that we want to add to the
             addlist or dellist
        """
        # Needed for instance check.
        from mapcore.planning.parsers.pddl_parser import Variable
        nextPredicate = None
        isNegative = False
        if c.key == 'not':
            # This is a negative effect, only one child allowed.
            if len(c.children) != 1:
                raise SemanticError('Error not statement with multiple '
                                    'children in effect of action')
            nextPredicate = c.children[0]
            isNegative = True
        else:
            nextPredicate = c
        # Check whether predicate was defined previously.
        if not nextPredicate.key in self._predicates:
            raise SemanticError('Error: unknown predicate %s used in effect '
                                'of action' % nextPredicate.key)
        if nextPredicate == None:
            raise SemanticError('Error: NoneType predicate used in effect of '
                                'action')
        predDef = self._predicates[nextPredicate.key]
        signature = list()
        count = 0
        # Check whether predicate is used with the correct signature.
        if len(nextPredicate.children) != len(predDef.signature):
            raise SemanticError('Error: wrong number of arguments for '
                                'predicate ' + nextPredicate.key +
                                ' in effect of action')
        # Apply to all parameters.
        for v in nextPredicate.children:
            if isinstance(v.key, Variable):
                signature.append((v.key.name, predDef.signature[count][1]))
            else:
                signature.append((v.key, predDef.signature[count][1]))
            count += 1

        # Add a new effect to the positive or negative effects respectively.
        if isNegative:
            effect.dellist.add(pddl.Predicate(nextPredicate.key, signature))
        else:
            effect.addlist.add(pddl.Predicate(nextPredicate.key, signature))

    def visit_effect_stmt(self, node):
        """ Visits a PDDL effect statement."""
        formula = node.formula
        effect = pddl.Effect()
        # For now we only allow 'and' in the effect.
        if formula.key == 'and':
            for c in formula.children:
                # Call helper.
                self.add_effect(effect, c)
        else:
            # Call helper.
            self.add_effect(effect, formula)
        # Store effect in node.
        self.set_in(node, effect)


class TraversePDDLProblem(PDDLVisitor):
    """The PDDL-problem Visitor.

    Expands the functionality of the PDDLVisitor to traversal of ASTs that
    represent a PDDL-problem file. It results in the PDDL data structure
    (pddl.py) representation of the problem file.
    """
    def get_in(self, node):
        """
        Helper method to access a global hash in which information for each
        node in the AST can be stored.
        """
        return self._nodeHash[node]

    def set_in(self, node, val):
        """
        Helper method to write a global hash in which information for each node
        in the AST can be stored.
        """
        self._nodeHash[node] = val

    def get_problem(self):
        """Getter for the resulting pddl-problem data structure."""
        return self._problemDef
    problemDef = property(get_problem)

    def __init__(self, domain):
        """Constructor for pddl-problem visitor.

        Keyword arguments:
        domain -- the coressponding pddl-domain datastrucutre
        """
        self._domain = domain
        self._nodeHash = dict()
        self._objects = dict()
        self._problemDef = None

    def visit_problem_def(self, node):
        """Visits a PDDL-problem definition."""
        # Check whether the in the problem file referenced domain name matches
        # the supplied domain data structure.
        if node.domainName != self._domain.name:
            raise SemanticError('Error trying to parse problem file with '
                                'domain: %s together with a domain file that '
                                'specifies domain: %s' %
                                (node.domainName, self._domain.name))
        # Apply to all object definitions.
        for o in node.objects:
            o.accept(self)

        # Apply to the initial state definition.
        node.init.accept(self)
        init_list = self.get_in(node.init)

        # Apply to the goal state definition.
        node.goal.accept(self)
        goal_list = self.get_in(node.goal)

        if node.constraints:
            node.constraints.predicates.extend(goal_list)
            node.constraints.init_predicates.extend(init_list)
            # Apply to the constraints state definition.
            node.constraints.accept(self)
            constraints_dict = self.get_in(node.constraints)
        else:
            constraints_dict = {}

        # Create the problem data structure.
        self._problemDef = pddl.Problem(node.name, self._domain, self._objects,
                                        init_list, goal_list, constraints_dict)

    def visit_object(self, node):
        """ Visits a PDDL-problem object definition."""
        type_def = None
        # Check for multiple definition of objects.
        if node.name in self._objects:
            raise SemanticError('Error multiple defines of object with name ' +
                                node.name)
        # Untyped objects get the standard type 'object'.
        if node.typeName == None:
            type_def = self._domain.types['object']
        else:
            # Check whether used type was introduced in domain file.
            if not node.typeName in self._domain.types:
                raise SemanticError('Error: unknown type ' + node.typeName +
                                    ' used in object definition!')
            type_def = self._domain.types[node.typeName]
        self._objects[node.name] = type_def

    def visit_init_stmt(self, node):
        """ Visits a PDDL-problem initial state statement."""
        initList = list()
        # Apply to all predicates in the statement.
        for p in node.predicates:
            p.accept(self)
            pred = self.get_in(p)
            initList.append(pred)
        self.set_in(node, initList)

    def add_goal(self, goal, c):
        """Helper function for visit_goal_stmt.

        Keyword arguments:
        goal -- a list of goals
        c -- a formula representing a goal we want to add to the goal list
        """
        # Check whether predicate was introduced in domain file.
        if not c.key in self._domain.predicates:
            raise SemanticError('Error: unknown predicate ' + c.key +
                                ' in goal definition')
        # Get predicate from the domain data structure.
        predDef = self._domain.predicates[c.key]
        signature = list()
        count = 0
        # Check whether the predicate uses the correct signature.
        if len(c.children) != len(predDef.signature):
            raise SemanticError('Error: wrong number of arguments for '
                                'predicate ' + c.key + ' in goal')
        # take agent from formula pred and role from domain.pred.definition
        for v in c.children:
            signature.append((v.key, predDef.signature[count][1]))
            count += 1
        # Add the predicate to the goal.
        goal.append(pddl.Predicate(c.key, signature))

    def visit_goal_stmt(self, node):
        """ Visits a PDDL-problem goal state statement."""
        formula = node.formula
        goal = list()
        # For now we only allow 'and' in the goal.
        if formula.key == 'and':
            for c in formula.children:
                if not isinstance(c.key, str):
                    raise SemanticError('Error predicate with non str key: ' +
                                        ''.join([c2.key.name + ' '
                                        for c2 in formula.children]))
                # Call helper.
                self.add_goal(goal, c)
        else:
            # Only a single predicate is allowed then (s.a.)
            if not formula.key in self._domain.predicates:
                raise SemanticError('Error: predicate in goal definition is '
                                    'not in CNF')
            # Call helper.
            self.add_goal(goal, formula)
        self.set_in(node, goal)

    def get_formula(self,formula, key):
        if not formula.key == key:
            formula = formula.children[0]
            if len(formula.children) > 1:
                for child in formula.children:
                    if child.key == key:
                        formula = child
            formula = self.get_formula(formula, key)
        return formula

    def blocks_blocknames(self, predicates, init_predicates, c, blocknames, blocks, constr):
        init_pred = [pred for pred in init_predicates if pred.name == c.key]
        for predicate in [predicate for predicate in predicates if predicate.name == c.key]:
            if predicate.signature[0][0] == c.children[0].key:
                keys = (c.children[0].key, predicate.signature[1][0])
                blocknames.setdefault(predicate.name, set()).add(keys)
                blocks.append(predicate.signature[1][0])
            elif not isinstance(c.children[0].key, str):
                keys = (predicate.signature[0][0], predicate.signature[1][0])
                blocknames.setdefault(predicate.name, set()).add(keys)
                blocks.append(predicate.signature[0][0])
            elif constr and init_pred:
                cognitive_preds = [pred for pred in init_pred if c.children[0].key == pred.signature[0][0]]
                for pred in cognitive_preds:
                    keys = (pred.signature[0][0], pred.signature[1][0])
                    preds = []
                    for con in constr:
                        if con.signature[0][0] == pred.signature[1][0]:
                            preds.extend([const for const in constr if const.signature[1][0]== con.signature[1][0]])
                    others = [(keys[0], pred.signature[0][0]) for pred in preds]
                    for key in others:
                        blocknames.setdefault(predicate.name, set()).add(key)
                break
        # return (blocks, blocknames)

    def visit_constraints_stmt(self, node):
        formula = node.formula
        constr = list()
        constraints = {}
        agents = {}
        changed = {}
        predicates = node.predicates
        init_predicates = node.init_predicates

        if formula.key == 'and':
            for child in formula.children:
                formula_for_changing = self.get_formula(child, 'forall')
                changed_key = formula_for_changing.children[0].children[1].key+formula_for_changing.children[0].key
                formula = self.get_formula(child, 'implies')
                blocks = []
                predDef = []
                agent = None
                for c in formula.children:
                    signature = []
                    sig = []
                    blocknames = {}
                    if c.key == "or":
                        for child in c.children:
                            predDef.append(self._domain.predicates[child.key])
                            self.blocks_blocknames(predicates, init_predicates, child, blocknames, blocks, constr)
                    else:
                        predDef.append(self._domain.predicates[c.key])
                        self.blocks_blocknames(predicates, init_predicates, c, blocknames, blocks, constr)


                    if len(blocknames):
                        unique = []
                        for predD in predDef:
                            for bltype in blocknames:
                                if predD.name == bltype:
                                    for name in blocknames.get(bltype):
                                        count = 0
                                        for v in name:
                                            sig.append((v, predD.signature[count][1]))
                                            count += 1
                                        sig.insert(0, predD.name)
                                        signature.append(sig)
                                        sig = []
                            if c.key == "or":

                                for child in c.children:
                                    for sign in signature:
                                        if child.key == sign[0]:
                                            predsign = [s for s in sign if not s == sign[0]]
                                            if predsign not in unique:
                                                constr.append(pddl.Predicate(child.key, predsign))
                                                unique.append(predsign)

                            else:
                                for sign in signature:
                                    if c.key == sign[0]:
                                        predsignature = [s for s in sign if not s == sign[0]]
                                        agent = predsignature[0][0]
                                        constr.append(pddl.Predicate(c.key, predsignature))
                        predDef = []

                    elif len(blocks):
                        # if holding predicate
                        for predD in predDef:
                            for block in blocks:
                                key = None
                                child_place = 0
                                for child in c.children:
                                    if isinstance(child.key, str):
                                        key = child.key
                                        child_place = c.children.index(child)
                                lists = [block]
                                lists.insert(child_place, key)
                                #lists = [c.children[0].key, block]
                                blocknames.setdefault(predD.name, []).append(lists)
                                agent = key
                                for name in blocknames.get(predD.name):
                                    count = 0
                                    for v in name:
                                        sig.append((v, predD.signature[count][1]))
                                        count += 1
                                    signature.append(sig)
                                    sig = []
                                for sign in signature:
                                    constr.append(pddl.Predicate(c.key, sign))
                                signature = []
                                blocknames = {}
                    else:
                        raise SemanticError('unknown predicate!')
                #constraints[agent] = constr
                changed[changed_key] = constr
                constraints.setdefault(agent, {}).update(changed)
                constr = []
                changed = {}



            self.set_in(node, constraints)




    def visit_predicate_instance(self, node):
        """ Visits a PDDL-problem predicate instance."""
        signature = list()
        # Visit all parameters.
        for o in node.parameters:
            o_type = None
            # Check whether predicate was introduced in objects or domain
            # constants.
            if not (o in self._objects or o in self._domain.constants):
                raise SemanticError('Error: object ' + o + ' referenced in '
                                    'problem definition - but not defined')
            elif o in self._objects:
                o_type = self._objects[o]
            elif o in self._domain.constants:
                o_type = self._domain.constants[o]
            signature.append((o, (o_type)))
        self.set_in(node, pddl.Predicate(node.name, signature))
