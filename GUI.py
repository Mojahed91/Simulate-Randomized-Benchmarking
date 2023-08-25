import tkinter as tk
from tkinter import IntVar, StringVar, Checkbutton, Label, Entry, Radiobutton, OptionMenu, Button
import matplotlib.pyplot as plt
import numpy
import numpy as np
import random
from scipy.optimize import curve_fit
from scipy.linalg import expm, sinm, cosm
from sympy import *
from tkinter import *
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Constants and Matrices
EPSILON = 0
INTERLEAVED = 1
NORMAL = 0

# Pauli Matrices
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]])
S_dag = S.conj().T

# Derived Gates
HS = S.dot(H)
S_dagH = H.dot(S_dag)
YH = H.dot(Y)
XH = H.dot(X)
ZH = H.dot(Z)
ZS = S.dot(Z)
YS = S.dot(Y)
XS = S.dot(X)
SHS = S.dot(H).dot(S)
ZS_dagH = H.dot(S_dag).dot(Z)
XS_dagH = H.dot(S_dag).dot(X)
YS_dagH = H.dot(S_dag).dot(Y)
YHS = Y.dot(H).dot(S)
ZHS = S.dot(H).dot(Z)
XHS = S.dot(H).dot(X)
XSHS = S.dot(H).dot(S).dot(X)
YSHS = S.dot(H).dot(S).dot(Y)
ZSHS = S.dot(H).dot(S).dot(Z)
T_gate = np.array([[0, 1], [0, 0]])

single_q_cliff = [X, Y, Z, I, H, S, HS, S_dagH, YH, XH, ZH, ZS,
                  YS, XS, SHS, ZS_dagH, XS_dagH, YS_dagH, YHS, ZHS, XHS, XSHS, YSHS, ZSHS]
Pauli_transfer_matrix = [I, HS, S_dagH]

# 2D 4x4 gates
cnot = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
mixed_state_noise = 0.25 * np.eye(4)
ground_state_noise = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
noisy_gate_matrix = np.kron(I, I)
pauli_Interleaved = X  # coherent error in irb method
pauli_periodicity = Z  # coherent error in Raam method

spont_matrix = np.array([[0, 1], [0, 0]]) / np.sqrt(2)
decoh_matrix = np.array([[1, 0], [0, -1]]) / np.sqrt(2)
spont_matrix_fix = np.array([[1, 0], [0, 0]]) / np.sqrt(2)
damping0_matrix = np.array([[1, 0], [0, np.sqrt(1-EPSILON)]])
damping1_matrix = np.array([[0, np.sqrt(EPSILON)], [0, 0]])

num_of_states = 4

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def create_label_entry(depolarized_noise_eq, text, default_val, row, col, col_span=1, fg=None):
    Label(depolarized_noise_eq, text=text, pady=5, fg=fg).grid(row=row, column=col)
    entry = Entry(depolarized_noise_eq, width=5)
    entry.grid(row=row, column=col + col_span)
    entry.insert(0, default_val)
    return entry


def create_radiobuttons(depolarized_noise_eq, options, var, row, col):
    for index, (text, val) in enumerate(options):
        Radiobutton(depolarized_noise_eq, text=text, pady=3, variable=var, value=val).grid(row=row + index, column=col)


def create_option_menu(depolarized_noise_eq, options, row, col):
    var = StringVar()
    var.set(options[0])
    dropdown = OptionMenu(depolarized_noise_eq, var, *options)
    dropdown.grid(row=row, column=col)
    return var


def GUI():
    depolarized_noise_eq = Tk()

    rb = IntVar()
    special_equation_button = Checkbutton(depolarized_noise_eq, text=" show interleaved RB   ", variable=rb, fg="red")
    special_equation_button.grid(row=11, column=5)

    label_cliff_len = Label(depolarized_noise_eq, text="clifford length", pady=5)
    label_cliff_len.grid(row=0, column=0)
    max_seq = Entry(depolarized_noise_eq, width=5)
    max_seq.grid(row=0, column=1)
    max_seq.insert(0, "100")

    label_theta = Label(depolarized_noise_eq, text="coherent error [theta]", pady=5)
    label_theta.grid(row=1, column=0)
    theta = Entry(depolarized_noise_eq, width=5)
    theta.grid(row=1, column=1)
    theta.insert(0, "0")

    label_p = Label(depolarized_noise_eq, text="P (Normal RB)", pady=5)
    label_p.grid(row=2, column=0)
    p_normal = Entry(depolarized_noise_eq, width=5)
    p_normal.grid(row=2, column=1)
    p_normal.insert(0, "0.998")

    label_p = Label(depolarized_noise_eq, text="P (Interleaved RB)", pady=5)
    label_p.grid(row=3, column=0)
    p_interleaved = Entry(depolarized_noise_eq, width=5)
    p_interleaved.grid(row=3, column=1)
    p_interleaved.insert(0, "0.998")

    label_shots = Label(depolarized_noise_eq, text="number of shots", pady=5)
    label_shots.grid(row=4, column=0)
    shots = Entry(depolarized_noise_eq, width=5)
    shots.grid(row=4, column=1)
    shots.insert(0, "10")

    RB_type = Label(depolarized_noise_eq, text="RB type", pady=5, fg="red")
    RB_type.grid(row=5, column=0)
    interleaved = IntVar()
    Normal_RB = Radiobutton(depolarized_noise_eq, text="Normal RB", pady=3, variable=interleaved, value=0)
    Interleaved_RB = Radiobutton(depolarized_noise_eq, text="Interleaved RB", pady=3, variable=interleaved, value=1)
    Normal_RB.grid(row=6, column=0)
    Interleaved_RB.grid(row=7, column=0)

    noise_type_label = Label(depolarized_noise_eq, text="noise type", pady=5, fg="red")
    noise_type_label.grid(row=8, column=0)
    noise_type = IntVar()
    button_ground = Radiobutton(depolarized_noise_eq, text="ground state noise", pady=3, variable=noise_type, value=0)
    button_mixed = Radiobutton(depolarized_noise_eq, text="mixed state noise ", pady=3, variable=noise_type, value=1)
    button_ground.grid(row=9, column=0)
    button_mixed.grid(row=10, column=0)

    num_of_cnot = Label(depolarized_noise_eq, text="number of CNOT in one unit", pady=5, fg="red")
    num_of_cnot.grid(row=12, column=0)
    num_of_cnots = IntVar()
    button_1cnot = Radiobutton(depolarized_noise_eq, text="1 cnot", pady=3, variable=num_of_cnots, value=0)
    button_2cnot = Radiobutton(depolarized_noise_eq, text="2 cnot", pady=3, variable=num_of_cnots, value=1)
    button_3cnot = Radiobutton(depolarized_noise_eq, text="3 cnot", pady=3, variable=num_of_cnots, value=2)
    button_4cnot = Radiobutton(depolarized_noise_eq, text="4 cnot", pady=3, variable=num_of_cnots, value=3)
    button_1cnot.grid(row=13, column=0)
    button_2cnot.grid(row=14, column=0)
    button_3cnot.grid(row=15, column=0)
    button_4cnot.grid(row=16, column=0)

    label_gates = Label(depolarized_noise_eq, text="Interleaved gate", pady=5, fg="red")
    gate_interleaved = StringVar()
    options_gate = [
        "CNOT",
        "X GATE",
        "Y GATE",
        "Z GATE",
        "H GATE",
        "SWAP",
        "S GATE",
    ]
    gate_interleaved.set(options_gate[0])
    drop = OptionMenu(depolarized_noise_eq, gate_interleaved, *options_gate)
    label_gates.grid(row=0, column=3)
    drop.grid(row=1, column=3)

    label_noisy_gate = Label(depolarized_noise_eq, text="Type of Quantum channel", pady=5, fg="red")
    noisy_gate = StringVar()
    options_noisy_gate = [
        "no noise",
        "spontaneous emission noise",
        "damping_noise",
        "dephasing",
    ]
    noisy_gate.set(options_noisy_gate[0])
    drop_noisy_gate = OptionMenu(depolarized_noise_eq, noisy_gate, *options_noisy_gate)
    label_noisy_gate.grid(row=2, column=3)
    drop_noisy_gate.grid(row=3, column=3)

    label_stringth_noise = Label(depolarized_noise_eq, text="strength of noisy matrix", pady=5)
    label_stringth_noise.grid(row=5, column=3)
    stringth_noise = Entry(depolarized_noise_eq, width=3)
    stringth_noise.grid(row=5, column=4)
    stringth_noise.insert(0, "0")

    scale_label = Label(depolarized_noise_eq, text="Plot y Scale", pady=5, fg="red")
    y_scale = IntVar()
    exp_scale = Radiobutton(depolarized_noise_eq, text="exponential scale", pady=3, variable=y_scale, value=0)
    log_scale_b = Radiobutton(depolarized_noise_eq, text="log scale             ", pady=3, variable=y_scale, value=1)

    scale_label.grid(row=10, column=3)
    exp_scale.grid(row=12, column=3)
    log_scale_b.grid(row=13, column=3)

    theta_ratio = IntVar()
    theta_ratio_button = Checkbutton(depolarized_noise_eq, text="plot theta vs ratio", variable=theta_ratio)
    theta_ratio_button.grid(row=1, column=5)

    p_ratio = IntVar()
    theta_ratio_button = Checkbutton(depolarized_noise_eq, text="p vs ratio noise    ", variable=p_ratio)
    theta_ratio_button.grid(row=2, column=5)
    # _______________________  add space start       ______________________________________________
    space = Label(depolarized_noise_eq, text="        ", pady=5)
    space.grid(row=0, column=4)
    # _______________________  add space end         ______________________________________________
    coherent_error_on = Label(depolarized_noise_eq, text="add coherent error", pady=5, fg="red")
    coherent_error_on.grid(row=4, column=5)
    coherent_on = IntVar()
    all_gates = Radiobutton(depolarized_noise_eq, text="on all gates", pady=3, variable=coherent_on, value=0)
    interleaved_only = Radiobutton(depolarized_noise_eq, text="on interleaved gate only", pady=3, variable=coherent_on, value=1)
    all_gates.grid(row=5, column=5)
    interleaved_only.grid(row=6, column=5)

    def run_program():

        global gat_intleav
        gat_intleav = irb_gate_type(gate_interleaved.get())[0]

        if int(rb.get()):
            if int(interleaved.get()):
                run(int(noise_type.get()), int(max_seq.get()),
                    float(p_interleaved.get()), float(p_normal.get()), float(theta.get()),
                    int(shots.get()), int(interleaved.get()), gate_interleaved.get(), int(y_scale.get()),
                    int(theta_ratio.get()), int(p_ratio.get()), noisy_gate.get(), float(stringth_noise.get())
                    , int(coherent_on.get()), int(num_of_cnots.get()))
            else:
                run(int(noise_type.get()), int(max_seq.get()),
                    0, float(p_normal.get()), float(theta.get()),
                    int(shots.get()), interleaved.get(), 0, int(y_scale.get()),
                    int(theta_ratio.get()), int(p_ratio.get()), noisy_gate.get(), float(stringth_noise.get())
                    , int(coherent_on.get()), int(num_of_cnots.get()))

    ok = Button(depolarized_noise_eq, text="             Ok               ", pady=5,
                command=run_program, fg="black", bg="blue")
    ok.grid(row=15, column=5)
    depolarized_noise_eq.mainloop()


def irb_gate_type(gate_interleaved):
    # Assuming I, X, Y, Z, H, S, cnot, and swap are predefined elsewhere in your code.
    gates = {
        "CNOT": cnot,
        "X GATE": np.kron(I, X),
        "Y GATE": np.kron(I, Y),
        "Z GATE": np.kron(I, Z),
        "H GATE": np.kron(I, H),
        "SWAP": swap,
        "S GATE": np.kron(I, S)
    }

    gate_selection_irb = gates.get(gate_interleaved, cnot)  # Default to cnot if gate name not found
    return [gate_selection_irb, gate_interleaved]


def noisy_matrix_type(gate_as_str, eps):
    I_4 = np.identity(4)

    # For "no noise"
    no_noise_matrix = np.identity(16)

    # For "spontaneous emission noise"
    A = np.kron(spont_matrix, I)
    A_dag = A.conj().T
    A_A_dag = np.dot(A, A_dag)
    I_A_A_dag = np.kron(I_4, A_A_dag)
    A_A_dag_I = np.kron(A_A_dag, I_4)
    spontaneous_emission_noise_matrix = eps * (np.kron(A, A_dag) - 0.5 * I_A_A_dag - 0.5 * A_A_dag_I) + np.identity(16)

    # For "damping_noise"
    damping0_matrix = np.array([[1, 0], [0, np.sqrt(1 - eps)]])
    damping1_matrix = np.array([[0, np.sqrt(eps)], [0, 0]])
    M0 = np.kron(damping0_matrix, numpy.conjugate(damping0_matrix))
    M1 = np.kron(damping1_matrix, numpy.conjugate(damping1_matrix))
    damping_noise_matrix = np.kron(M0 + M1, M0 + M1)

    # For "dephasing"
    dephasing0 = np.array([[np.sqrt(1 - eps), 0], [0, np.sqrt(1 - eps)]])
    dephasing1 = np.array([[np.sqrt(eps), 0], [0, 0]])
    dephasing2 = np.array([[0, 0], [0, np.sqrt(eps)]])
    M0 = np.kron(dephasing0, numpy.conjugate(dephasing0))
    M1 = np.kron(dephasing1, numpy.conjugate(dephasing1))
    M2 = np.kron(dephasing2, numpy.conjugate(dephasing2))
    dephasing_matrix = np.kron(M0 + M1 + M2, M0 + M1 + M2)

    noise_mappings = {
        "no noise": no_noise_matrix,
        "spontaneous emission noise": spontaneous_emission_noise_matrix,
        "damping_noise": damping_noise_matrix,
        "dephasing": dephasing_matrix,
    }

    return noise_mappings.get(gate_as_str, np.kron(I, I))


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def get_random_gate():
    r = random.randint(1, 11520)
    r1_24 = random.randint(1, 24)
    r2_24 = random.randint(1, 24)
    clifford = np.kron(single_q_cliff[r1_24 - 1], single_q_cliff[r2_24 - 1])
    r_3 = random.randint(1, 3)

    if r <= 576:  # 24 * 24  single_q_cliff tensor single_q_cliff
        return clifford
    elif r > 576 & r <= 1152:  # 24 * 24 and SWAP
        return np.dot(clifford, swap)
    elif r > 1152 & r <= 6336:  # 24 * 24 with  CNOT and [Pauli transfer matrix = {I ,Rs , Rs^2}]
        return np.array(np.dot(clifford, cnot), Pauli_transfer_matrix[r_3-1])
    elif r > 6336 & r <= 11520:  # 24  * 24 with  ISWAP and with Pauli transfer matrix
        return np.array(np.dot(clifford, iswap), Pauli_transfer_matrix[r_3 - 1])


def evolve_state_with_noise(liouville_u, rho_before, p, noise_matrix):
    """
    Evolves a quantum state with potential noise.

    Args:
    - liouville_u (np.ndarray): The Liouvillian representation of the unitary evolution.
    - rho_before (np.ndarray): The density matrix representing the initial quantum state.
    - p (float): The probability of the quantum system evolving without noise.
    - noise_matrix (np.ndarray): The matrix representation of the quantum noise.

    Returns:
    - np.ndarray: The evolved quantum state.
    """

    # Evolution without noise
    unitary_evolution = np.dot(liouville_u, rho_before)

    # Weighted combination of noise-free evolution and noisy evolution
    rho_after = p * unitary_evolution + (1 - p) * noise_matrix

    return rho_after


def evolve_state_with_noisy_gate(liouville_u, rho_before, p, noise_matrix, gate_type):
    """
    Evolves a quantum state using a unitary operation, a noisy gate, and a noise model.

    Args:
    - liouville_u (np.ndarray): The Liouvillian representation of the unitary evolution.
    - rho_before (np.ndarray): The density matrix representing the initial quantum state.
    - p (float): Probability of the quantum system undergoing the intended noisy gate operation.
    - noise_matrix (np.ndarray): Matrix representation of an inherent noise model.
    - gate_type (any type): This argument is accepted but not used in the current function.

    Returns:
    - np.ndarray: The evolved quantum state.
    """

    global noisy_gate_matrix

    # Evolution by applying the unitary and then the noisy gate
    unitary_evolution = np.dot(liouville_u, rho_before)
    noisy_evolution = np.dot(noisy_gate_matrix, unitary_evolution)

    # Weighted combination of noisy gate evolution and inherent noise evolution
    rho_after = p * noisy_evolution + (1 - p) * noise_matrix

    return rho_after


def rho_next_correct_inv(liouville_u, rho):
    return np.dot(liouville_u, rho)


def rho_next_correct(liouville_u, rho):
    u_rho_u_d = np.dot(liouville_u, rho)
    return u_rho_u_d


def u_total(u_before, u_new):
    return np.dot(u_new, u_before)


def convert_density_m_to_liouville_vector(rho_matrix):
    return rho_matrix.ravel()


def convert_unitary_m_to_liouville_matrix(unitary_matrix, type_gate):
    global pauli_Interleaved, theta_coherent
    if coherent_error_on == NORMAL or (type_gate == INTERLEAVED and coherent_error_on == INTERLEAVED):
        hamiltonian = np.dot(1j * theta_coherent, np.kron(pauli_Interleaved, pauli_Interleaved))
        unitary_matrix = np.dot(expm(hamiltonian),unitary_matrix)
        liouville_matrix = np.kron(unitary_matrix, numpy.conjugate(unitary_matrix))
        return liouville_matrix

    return np.kron(unitary_matrix, numpy.conjugate(unitary_matrix))


def convert_vector_to_matrix(vector, dim_num_states):
    return np.reshape(vector, (dim_num_states, dim_num_states))


def random_series_multiplication(noise_matrix, m, rho_initial):
    global noisy_gate_matrix
    global coherent_error_on, gate_irb, num_of_cnot
    rho_final = convert_density_m_to_liouville_vector(rho_initial)
    rho_correct = convert_density_m_to_liouville_vector(rho_initial)
    u_tot = np.identity(num_of_states)
    flattened_noise_matrix = convert_density_m_to_liouville_vector(noise_matrix)
    if interleave:
        liouville_gat_interleaved = convert_unitary_m_to_liouville_matrix(gate_irb, INTERLEAVED)

        for i in range(int(m)):
            random_matrix = get_random_gate()
            liouville_random_matrix = convert_unitary_m_to_liouville_matrix(random_matrix, NORMAL)
            u_tot = u_total(u_tot, np.dot(random_matrix, cnot))
            rho_correct = rho_next_correct(liouville_random_matrix, rho_correct)
            for h in range(num_of_cnot):
                rho_correct = rho_next_correct(liouville_gat_interleaved, rho_correct)

            rho_final = evolve_state_with_noisy_gate(liouville_random_matrix, rho_final, p1, flattened_noise_matrix, NORMAL)
            for h in range(num_of_cnot):
                rho_final = evolve_state_with_noisy_gate(liouville_gat_interleaved, rho_final, p2, flattened_noise_matrix, INTERLEAVED)

    else:
        for i in range(m):
            random_matrix = get_random_gate()
            liouville_random_matrix = convert_unitary_m_to_liouville_matrix(random_matrix, NORMAL)
            u_tot = u_total(u_tot, random_matrix)
            rho_correct = rho_next_correct(liouville_random_matrix, rho_correct)
            rho_final = evolve_state_with_noisy_gate(liouville_random_matrix, rho_final, p1, flattened_noise_matrix, NORMAL)

    u_inv = np.linalg.inv(u_tot)
    liouville_u_inv = convert_unitary_m_to_liouville_matrix(u_inv, NORMAL)
    rho_correct = rho_next_correct_inv(liouville_u_inv, rho_correct)
    rho_final = evolve_state_with_noise(liouville_u_inv, rho_final, p1, flattened_noise_matrix)

    rho_final = convert_vector_to_matrix(rho_final, num_of_states)
    trac = np.trace(rho_final)
    rho_final = np.dot(1/trac, rho_final)
    rho_correct = convert_vector_to_matrix(rho_correct, num_of_states)
    return [rho_final, rho_correct]


def survival_probability(series_m):
    if log_scale:
        return np.trace(np.dot(series_m[1], series_m[0])) - 0.25
    else:
        return np.trace(np.dot(series_m[1], series_m[0]))


def epc(alpha, num_q):
    return ((pow(2, num_q) - 1) / (pow(2, num_q))) * (1 - alpha)


def noise_model_simulation(seq_length, noise_matrix, rho_initial):
    survival_prob_vector = [] * seq_length
    for seq_l in range(seq_length):
        series_m = random_series_multiplication(noise_matrix, seq_l, rho_initial)
        survival_prob = survival_probability(series_m)
        survival_prob_vector.append(survival_prob.real)

    return survival_prob_vector


def noise_model_repetition(noise_matrix, rho_initial, rep_num):
    seq_vector = np.arange(1, max_seq_length + 1)
    survival_prob_rep_vector = [[]] * rep_num

    for rep in range(rep_num):
        survival_prob_vector = noise_model_simulation(max_seq_length, noise_matrix, rho_initial)
        survival_prob_rep_vector[rep] = survival_prob_vector

    return [seq_vector, survival_prob_rep_vector]


def fitted_function(x, a, b, c):
    f = np.dot(pow(b, x), a) + 0.25
    if log_scale:
        f = f - 0.25
    return f


def fit_and_plot_data(seq_vector, survival_prob_rep_vector, rep_num, max_seq_length, ax, p_val, clr, str_type,
                      yes_plot):
    clifford_series = seq_vector
    survival_prob = np.array(survival_prob_rep_vector).mean(axis=0)

    survival_prob = survival_prob.data
    if yes_plot:
        for rep in range(rep_num):
            ax.plot(clifford_series, survival_prob_rep_vector[rep], 'x')
    param, extras = curve_fit(fitted_function, clifford_series, survival_prob)

    a, alpha, c = param
    xx = np.linspace(0, max_seq_length, max_seq_length)
    if yes_plot:
        ax.plot(clifford_series, fitted_function(xx, a, alpha, c), '-', color=clr,
                label=str_type)

    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('Gate Depth')
    return alpha


def print_matrix(a, name):
    print(name + " = " + str(a[0]))
    print("   " + " " * len(name) + str(a[1]))


def number_input(number):
    current_value = e.get()
    e.delte(0, END)
    e.insert(0, str(number) + str(current_value))


def plot_ratio_thetas(ax, thetas, ratio, p, shots, sample):
    for i in range(sample):
        ax.plot(thetas, ratio[i], 'x')
        print(ratio[i])

    ax.set_title(" theta vs deviation from fitted alpha")
    ax.set_ylabel('deviation  =  input (p) - fitted (alpha) ')
    ax.set_xlabel('theta')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.text((x_min + x_max) / 4, (y_min + y_max) / 2,
            "p = " + str(p) + "\n" +
            "shots = " + str(shots) + "\n",
            style='italic', fontsize=12, bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})


def ratio_error(alpha, p):
    return alpha - p


def p_vs_ratio(ax, noise_matrix, rho_initial, rep_num, gate_depth):
    no_plot = 0
    ps = [0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.89, 0.9, 0.91,
          0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999]
    samples_of_p = len(ps)

    ratio = numpy.zeros(samples_of_p)

    for j in range(samples_of_p):
        p1 = ps[j]
        model_nrb = noise_model_repetition(noise_matrix, rho_initial, rep_num)
        alpha_nrb = fit_and_plot_data(model_nrb[0], model_nrb[1], rep_num, max_seq_length, ax, ps[j], 'b', 'no',
                                      no_plot)
        ratio[j] = (1 - p1) / (1 - alpha_nrb)

    ax.plot(ps, ratio, 'x')

    ax.set_title("Error per gat (input p) vs  (1-p) / (1 - alpha(fitted parameter))")
    ax.set_ylabel('(1-p) / (1 - alpha)')
    ax.set_xlabel('Error per gat (input p)')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.text(x_min, (y_min + y_max) * 0.4,
            "Single shots (Without Averaging)\n" + "Gate Depth = " + str(gate_depth) + "\n",
            style='italic', fontsize=12,
            bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})


def delta_theta_vs_ratio(ax, noise_matrix, rho_initial, rep_num, p):
    global e
    global e_
    global p1

    no_plot = 0
    ps = [0.9, 0.98, 0.99, 0.999, 0.9999]
    samples_of_p = len(ps)

    thetas = [0.00001, 0.0001, 0.001,
              0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
              0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
              0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
    ratio = numpy.zeros(shape=(samples_of_p, len(thetas)))

    for j in range(samples_of_p):
        p1 = ps[j]
        for i in range(len(thetas)):
            e_ = expm(1j * thetas[i] * np.kron(Z, Z))
            e = expm(-1j * thetas[i] * np.kron(Z, Z))
            model_nrb = noise_model_repetition(noise_matrix, rho_initial, rep_num)
            alpha_nrb = fit_and_plot_data(model_nrb[0], model_nrb[1], rep_num, max_seq_length, ax, ps[j], 'b', 'no',
                                          no_plot)
            ratio[j, i] = p1 - alpha_nrb

    plot_ratio_thetas(ax, thetas, ratio, p, rep_num, samples_of_p)


def show_details_irb(epc_nrb, epc_irb, shots, alpha_irb, alpha_nrb, theta, noise, ax):
    if log_scale:
        plt.yscale("log")
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.text((x_min + x_max) / 3, y_min,
            noise + "\n" +
            "EPC_RB(" + str(round(1 - p1, 2) * 100) + "%) = " + epc_nrb + "\n" +
            "EPC_IRB(" + str(round(1 - p2, 2) * 100) + "%) = " + epc_irb + "\n" +
            "\u03B1_rb = " + str(alpha_nrb) + "  (fitted)\n" +
            "\u03B1_irb = " + str(alpha_irb) + "  (fitted)\n" +
            "p_rb = " + str(p1) + "  (input)\n" +
            "p_irb = " + str(p2) + "  (input)\n" +
            "shots = " + str(shots) + "\n" +
            "\u03F4 = " + str(theta) + "\n"
            , style='italic', fontsize=12, bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    ax.set_title("Simulation the Depolarized Noise Equation               ")


def show_details_nrb(p, epc_nrb, model_nrb, shots, alpha_nrb, theta, noise, ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.text((x_min + x_max) / 3, y_min,
            noise + "\n" +
            "EPC(" + str(round(1 - p, 2) * 100) + "%) = " + epc_nrb + "\n" +
            "\u03B1 = " + str(alpha_nrb) + "  (fitted)\n" +
            "P = " + str(p) + "  (input)\n" +
            "1-P\\1-\u03B1 = " + str((1 - p) / (1 - alpha_nrb)) + "\n" +
            "shots = " + str(shots) + "\n" +
            "\u03F4 = " + str(theta) + "\n"
            , style='italic', fontsize=12, bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    ax.set_title("Simulation the Depolarized Noise Equation               ")


def run(noise_type, max_seq, p_interleaved, p_normal, theta, shots, interleaved, gate_interleaved,
        y_scale, showThtavsRatio, showPvsRatio, noisy_gate_, stringth_noise_, coherent_on, num_cnot):
    global e, e_, p1, interleave, max_seq_length, log_scale, rep_num, noisy_gate_matrix, \
        eps, coherent_error_on, num_of_cnot, theta_coherent

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()
    if noise_type == 0:
        noise_matrix = ground_state_noise
        noise = "ground_state_noise"
    if noise_type == 1:
        noise_matrix = mixed_state_noise
        noise = "mixed_state_noise"

    num_of_cnot = num_cnot + 1
    coherent_error_on = coherent_on
    eps = stringth_noise_
    interleave = interleaved
    p1 = p_normal
    max_seq_length = max_seq
    rep_num = shots
    log_scale = y_scale
    noisy_gate_matrix = noisy_matrix_type(noisy_gate_, eps)
    theta_coherent = theta
    # I_ = np.identity(num_of_states * 2)
    e_ = expm(1j * theta_coherent * pauli_Interleaved)
    e = expm(-1j * theta_coherent * pauli_Interleaved)

    rho_initial = np.kron([[1, 0], [0, 0]], [[1, 0], [0, 0]])

    yes_plot = 1

    if interleave:
        global p2, gate_irb, str_gate_irb
        p2 = p_interleaved
        p = p_interleaved
        gate_irb = irb_gate_type(gate_interleaved)[0]
        str_gate_irb = irb_gate_type(gate_interleaved)[1]
        str_type = "Interleaved RB with " + str_gate_irb
        model_irb = noise_model_repetition(noise_matrix, rho_initial, rep_num)
        alpha_irb = fit_and_plot_data(model_irb[0], model_irb[1], rep_num,
                                      max_seq_length, ax, p, 'r', str_type, yes_plot)
        epc_irb = str(epc(alpha_irb, 2))

        interleave = 0
        p = p_normal
        str_type = "normal RB"
        model_nrb = noise_model_repetition(noise_matrix, rho_initial, rep_num)
        alpha_nrb = fit_and_plot_data(model_nrb[0], model_nrb[1], rep_num,
                                      max_seq_length, ax, p, 'b', str_type, yes_plot)
        epc_nrb = str(epc(alpha_nrb, 2))
        show_details_irb(epc_nrb, epc_irb, shots, alpha_irb, alpha_nrb, theta, noise, ax)

        op = alpha_irb/alpha_nrb

        print("\n\n\n _    interleave    IRB")
        print("alpha = " + str(op))
        print("2EPC = " + str(2 * epc(op, 2)))
        print("EPC = " + str(epc(op, 2)))
    else:
        p = p_normal
        str_type = "normal RB"
        model_nrb = noise_model_repetition(noise_matrix, rho_initial, rep_num)
        alpha_nrb = fit_and_plot_data(model_nrb[0], model_nrb[1], rep_num, max_seq_length, ax, p, 'b', str_type,
                                      yes_plot)
        epc_nrb = str(epc(alpha_nrb, 2))

        show_details_nrb(p, epc_nrb, model_nrb, shots, alpha_nrb, theta, noise, ax)

    plt.legend("p = 0.99", loc="upper right")
    if log_scale:
        plt.yscale("log")
    elif showThtavsRatio:
        fig_theta = plt.figure(figsize=(12, 8))
        axx = fig_theta.add_subplot()
        delta_theta_vs_ratio(axx, noise_matrix, rho_initial, rep_num, p1)
    elif showPvsRatio:
        fig_p = plt.figure(figsize=(12, 8))
        axxx = fig_p.add_subplot()
        p_vs_ratio(axxx, noise_matrix, rho_initial, rep_num, max_seq_length)

    plt.show()


if __name__ == '__main__':
     GUI()
