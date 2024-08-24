# Module that calculates all possibles 765 states in TicTacToe and performs transformation of some game possition to
# equivalent state

rotation_order = [6, 3, 0, 7, 4, 1, 8, 5, 2] # clockwise
rotation_order_ccw = [2, 5, 8, 1, 4, 7, 0, 3, 6] # contra clockwise
def rotate(state_or_action, conterclockwise=False):
    order = rotation_order_ccw if conterclockwise else rotation_order
    if type(state_or_action) == int:
        action = state_or_action
        new_action = rotation_order[action]
        return new_action
    else:
        state = state_or_action
        new_state = []
        for i in range(9):
            new_state.append(state[order[i]])
        return new_state


reflection_order = [2, 1, 0, 5, 4, 3, 8, 7, 6]
def reflect(state_or_action):
    if type(state_or_action) == int:
        action = state_or_action
        new_action = reflection_order[action]
        return new_action
    else:
        state = state_or_action
        new_state = []
        for i in range(9):
            new_state.append(state[reflection_order[i]])
        return new_state

max_steps_n = 9


def get_equivalent(states, state):
    if len(states) == 0:
        return None
    translation = [0, 0]
    if state in states:
        return state, translation
    for i in range(3):
        state = rotate(state)
        translation[1] = i + 1
        if state in states:
            return state, translation
    state = reflect(rotate(state))
    translation = [1, 0]
    if state in states:
        return state, translation
    for i in range(3):
        state = rotate(state)
        translation[1] = i + 1
        if state in states:
            return state, translation
    return None # if no equivalent in states


def get_initial(state_or_action, translation):
    for i in range(translation[1]):
        state_or_action = rotate(state_or_action, conterclockwise=True)
    if translation[0] == 1:
        state_or_action = reflect(state_or_action)
    return state_or_action


def visualise_state(state):
    output_str = ""
    for i in range(3):
        for j in range(3):
            output_str+=str(state[i * 3 +j]) + " "
        output_str+="\n"
    print(output_str)


def get_possible_steps(state):
    possible_steps = []
    for i, value in enumerate(state):
        if value == 0:
            possible_steps.append(i)
    return possible_steps


def check_winning_condition(state):
    for i in range(3):
        if state[i * 3] != 0 and state[i * 3] == state[i * 3 + 1] and state[i * 3] == state[i * 3 + 2]:
            return True
        if state[0 * 3 + i] != 0 and state[0 * 3 + i] == state[1 * 3 + i] and state[0 * 3 + i] == state[2 * 3 + i]:
            return True
    if state[0] != 0 and (state[0] == state[1 * 3 + 1] == state[2 * 3 + 2]):
        return True
    if state[2] != 0 and state[2] == state[1 * 3 + 1] == state[2 * 3 + 0]:
        return True
    return False

def calculate_states(states, possible_steps_list, state, step_iter):
    who_steps = 1 if step_iter % 2 == 1 else 2

    possible_steps = get_possible_steps(state)
    for step_pos in possible_steps:
        new_state = state.copy()
        new_state[step_pos] = who_steps

        eqv_state = get_equivalent(states, new_state)
        if eqv_state == None:
            states.append(new_state)
            possible_steps_list.append(get_possible_steps(new_state))
            if check_winning_condition(new_state):
                return
            calculate_states(states, possible_steps_list, new_state, step_iter+1)


def calculate_states_and_actions():
    initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    states = []
    step_iter = 1
    actions = []
    calculate_states(states, actions, initial_state, step_iter)
    states.append(initial_state)
    actions.append([0, 1, 2, 3, 4, 5, 6, 7, 8])
    return states, actions

