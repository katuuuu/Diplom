import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

AU_KM = 149_597_870.7


@dataclass
class SystemParams:
    name: str
    planet: str
    satellite: str

    m1_over_m2: float
    eta: float
    a1_au: float
    a2_km: float

    e1: float
    e2: float

    g2: float = 0.0
    theta1_0: float = 0.0
    theta2_0: float = 0.0

    @property
    def mu(self) -> float:
        return self.m1_over_m2 / (1.0 + self.eta)

    @property
    def lambda_(self) -> float:
        return self.a1_au * AU_KM / self.a2_km

    @property
    def nu(self) -> float:
        return math.sqrt(self.mu / (self.lambda_ ** 3))

    @property
    def xi(self) -> float:
        r23 = self.a2_km
        r12 = self.a1_au * AU_KM

        return self.m1_over_m2 * (r23 / r12) ** 3


def build_system_catalog() -> list[SystemParams]:
    return [
        SystemParams("Земля — Луна", "Земля", "Луна",
                     332_946.0, 12_300e-6, 1.0, 383.4e3,
                     e1=0.0167, e2=0.0549),
        SystemParams("Марс — Фобос", "Марс", "Фобос",
                     332_946.0 / 0.107, 0.02e-6, 1.524, 9.38e3,
                     e1=0.0934, e2=0.0151),

        SystemParams("Марс — Деймос", "Марс", "Деймос",
                     332_946.0 / 0.107, 0.003e-6, 1.524, 23.46e3,
                     e1=0.0934, e2=0.00033),

        SystemParams("Юпитер — Ганимед", "Юпитер", "Ганимед",
                     332_946.0 / 317.8, 79e-6, 5.2, 1070e3,
                     e1=0.0489, e2=0.0013),

        SystemParams("Юпитер — Каллисто", "Юпитер", "Каллисто",
                     332_946.0 / 317.8, 58e-6, 5.2, 1883e3,
                     e1=0.0489, e2=0.0074),

        SystemParams("Юпитер — Ио", "Юпитер", "Ио",
                     332_946.0 / 317.8, 47e-6, 5.2, 422e3,
                     e1=0.0489, e2=0.0041),

        SystemParams("Юпитер — Европа", "Юпитер", "Европа",
                     332_946.0 / 317.8, 25e-6, 5.2, 671e3,
                     e1=0.0489, e2=0.0094),

        SystemParams("Сатурн — Титан", "Сатурн", "Титан",
                     332_946.0 / 95.16, 240e-6, 9.54, 1222e3,
                     e1=0.0565, e2=0.0288),

        SystemParams("Сатурн — Рея", "Сатурн", "Рея",
                     332_946.0 / 95.16, 4.1e-6, 9.54, 527e3,
                     e1=0.0565, e2=0.001),

        SystemParams("Сатурн — Япет", "Сатурн", "Япет",
                     332_946.0 / 95.16, 3.4e-6, 9.54, 3561e3,
                     e1=0.0565, e2=0.0286),

        SystemParams("Сатурн — Диона", "Сатурн", "Диона",
                     332_946.0 / 95.16, 1.9e-6, 9.54, 377e3,
                     e1=0.0565, e2=0.0022),

        SystemParams("Сатурн — Тефия", "Сатурн", "Тефия",
                     332_946.0 / 95.16, 1.09e-6, 9.54, 294.6e3,
                     e1=0.0565, e2=0.0001),

        SystemParams("Сатурн — Энцелад", "Сатурн", "Энцелад",
                     332_946.0 / 95.16, 0.19e-6, 9.54, 238e3,
                     e1=0.0565, e2=0.0047),

        SystemParams("Сатурн — Мимас", "Сатурн", "Мимас",
                     332_946.0 / 95.16, 0.07e-6, 9.54, 185.4e3,
                     e1=0.0565, e2=0.0196),

        SystemParams("Уран — Титания", "Уран", "Титания",
                     332_946.0 / 14.37, 40e-6, 19.19, 436e3,
                     e1=0.0463, e2=0.0011),

        SystemParams("Уран — Оберон", "Уран", "Оберон",
                     332_946.0 / 14.37, 35e-6, 19.19, 584e3,
                     e1=0.0463, e2=0.0014),

        SystemParams("Уран — Ариэль", "Уран", "Ариэль",
                     332_946.0 / 14.37, 16e-6, 19.19, 191e3,
                     e1=0.0463, e2=0.0012),

        SystemParams("Уран — Умбриэль", "Уран", "Умбриэль",
                     332_946.0 / 14.37, 13.49e-6, 19.19, 266.3e3,
                     e1=0.0463, e2=0.0039),

        SystemParams("Уран — Миранда", "Уран", "Миранда",
                     332_946.0 / 14.37, 0.75e-6, 19.19, 129.4e3,
                     e1=0.0463, e2=0.0013),

        SystemParams("Нептун — Тритон", "Нептун", "Тритон",
                     332_946.0 / 17.15, 210e-6, 30.07, 355e3,
                     e1=0.009, e2=0.000016),

        SystemParams("Нептун — Протей", "Нептун", "Протей",
                     332_946.0 / 17.15, 0.48e-6, 30.07, 118e3,
                     e1=0.009, e2=0.0005),

        SystemParams("Нептун — Нереида", "Нептун", "Нереида",
                     332_946.0 / 17.15, 0.29e-6, 30.07, 5513e3,
                     e1=0.009, e2=0.7507),
    ]

def choose_system() -> SystemParams:
    import tkinter as tk
    from tkinter import ttk, messagebox

    systems = build_system_catalog()
    selected_system = {"value": None}

    def submit():
        choice = combo.current()

        if choice == -1:
            messagebox.showerror("Ошибка", "Выберите систему из списка.")
            return

        selected_system["value"] = systems[choice]
        root.destroy()

    root = tk.Tk()
    root.title("Выбор системы")
    root.geometry("620x360")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use("clam")

    style.configure(
        "Title.TLabel",
        font=("Segoe UI", 18, "bold"),
        foreground="#1f2937"
    )

    style.configure(
        "Subtitle.TLabel",
        font=("Segoe UI", 10),
        foreground="#6b7280"
    )

    style.configure(
        "Main.TButton",
        font=("Segoe UI", 11, "bold"),
        padding=8
    )

    container = ttk.Frame(root, padding=25)
    container.pack(fill="both", expand=True)

    title = ttk.Label(
        container,
        text="Моделирование орбитального движения",
        style="Title.TLabel"
    )
    title.pack(anchor="w")

    subtitle = ttk.Label(
        container,
        text="Выберите систему «планета — спутник» для численного интегрирования",
        style="Subtitle.TLabel"
    )
    subtitle.pack(anchor="w", pady=(5, 25))

    card = ttk.Frame(container, padding=20, relief="ridge")
    card.pack(fill="x")

    label = ttk.Label(
        card,
        text="Система:",
        font=("Segoe UI", 11, "bold")
    )
    label.pack(anchor="w", pady=(0, 8))

    values = [
        f"{i}. {system.name}"
        for i, system in enumerate(systems, start=1)
    ]

    combo = ttk.Combobox(
        card,
        values=values,
        state="readonly",
        font=("Segoe UI", 11),
        width=50
    )
    combo.pack(fill="x")
    combo.current(0)

    info_label = ttk.Label(
        card,
        text="После выбора система будет использована для расчёта возмущённого и невозмущённого движения.",
        style="Subtitle.TLabel",
        wraplength=520
    )
    info_label.pack(anchor="w", pady=(12, 0))

    button = tk.Button(
        container,
        text="Начать расчёт",
        font=("Segoe UI", 12, "bold"),
        bg="#d0d0d0",
        fg="#111827",
        activebackground="#9ca3af",
        activeforeground="#111827",
        relief="flat",
        borderwidth=0,
        padx=12,
        pady=14,
        cursor="hand2",
        width=12,
        command=submit
    )

    button.pack(pady=25, ipady=6)

    root.bind("<Return>", lambda event: submit())
    root.mainloop()

    if selected_system["value"] is None:
        raise RuntimeError("Система не была выбрана.")

    return selected_system["value"]



def planet_position(theta1: float, params: SystemParams):
    e1 = params.e1
    lam = params.lambda_

    r1 = lam * (1.0 - e1**2) / (1.0 + e1 * math.cos(theta1))
    x1 = r1 * math.cos(theta1)
    y1 = r1 * math.sin(theta1)

    return x1, y1, r1


def theta1_derivative(theta1: float, params: SystemParams) -> float:
    e1 = params.e1
    numerator = (1.0 + e1 * math.cos(theta1)) ** 2
    denominator = (1.0 - e1**2) ** 1.5

    return 2.0 * math.pi * params.nu * numerator / denominator


def satellite_initial_conditions(params: SystemParams) -> np.ndarray:
    e2 = params.e2
    g2 = params.g2
    theta2 = params.theta2_0

    rho = (1.0 - e2**2) / (1.0 + e2 * math.cos(theta2))
    phi = g2 + theta2

    x0 = rho * math.cos(phi)
    y0 = -rho * math.sin(phi)

    theta2_prime = (
        2.0
        * math.pi
        * (1.0 + e2 * math.cos(theta2)) ** 2
        / ((1.0 - e2**2) ** 1.5)
    )

    drho_dtheta = (1.0 - e2**2) * e2 * math.sin(theta2) / (
        (1.0 + e2 * math.cos(theta2)) ** 2
    )

    rho_prime = drho_dtheta * theta2_prime

    u0 = rho_prime * math.cos(phi) - rho * math.sin(phi) * theta2_prime
    v0 = -rho_prime * math.sin(phi) - rho * math.cos(phi) * theta2_prime

    return np.array([x0, y0, u0, v0, params.theta1_0], dtype=float)


def rhs(
    tau: float,
    q: np.ndarray,
    params: SystemParams,
    perturbation: bool = True
) -> np.ndarray:
    x, y, u, v, theta1 = q

    r = math.sqrt(x**2 + y**2)

    if r < 1e-12:
        raise ValueError("Слишком малое расстояние r. Деление на ноль.")

    # Основное притяжение планеты
    du = -4.0 * math.pi**2 * x / r**3
    dv = -4.0 * math.pi**2 * y / r**3

    # Солнечное возмущение
    if perturbation:
        x1, y1, r1 = planet_position(theta1, params)
        dot = x1 * x + y1 * y
        mu = params.mu

        du += (
            -4.0 * math.pi**2 * mu * x / r1**3
            + 12.0 * math.pi**2 * mu * dot * x1 / r1**5
        )

        dv += (
            -4.0 * math.pi**2 * mu * y / r1**3
            + 12.0 * math.pi**2 * mu * dot * y1 / r1**5
        )

    dtheta1 = theta1_derivative(theta1, params)

    return np.array([u, v, du, dv, dtheta1], dtype=float)


def rk4_step(
    tau: float,
    q: np.ndarray,
    h: float,
    params: SystemParams,
    perturbation: bool = True
) -> np.ndarray:
    k1 = rhs(tau, q, params, perturbation)
    k2 = rhs(tau + h / 2.0, q + h * k1 / 2.0, params, perturbation)
    k3 = rhs(tau + h / 2.0, q + h * k2 / 2.0, params, perturbation)
    k4 = rhs(tau + h, q + h * k3, params, perturbation)

    return q + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate(params: SystemParams, tau_max: float, h: float, perturbation: bool = True):
    if h <= 0:
        raise ValueError("Шаг интегрирования h должен быть положительным.")

    steps = int(tau_max / h) + 1

    tau_values = np.zeros(steps)
    q_values = np.zeros((steps, 5))

    q = satellite_initial_conditions(params)

    tau_values[0] = 0.0
    q_values[0] = q

    tau = 0.0

    for i in range(1, steps):
        q = rk4_step(tau, q, h, params, perturbation)
        tau += h

        tau_values[i] = tau
        q_values[i] = q

    return tau_values, q_values

def plot_deviation_trajectory(
    q_perturbed: np.ndarray,
    q_unperturbed: np.ndarray,
    params: SystemParams
):
    x_pert = q_perturbed[:, 0]
    y_pert = q_perturbed[:, 1]

    x_unpert = q_unperturbed[:, 0]
    y_unpert = q_unperturbed[:, 1]

    # Отклонения
    dx = x_pert - x_unpert
    dy = y_pert - y_unpert

    # Перевод в километры
    dx_km = dx * params.a2_km
    dy_km = dy * params.a2_km

    plt.figure(figsize=(7, 7))

    plt.plot(dx_km, dy_km)

    plt.scatter(
        [0],
        [0],
        marker="o",
        label="Начальное совпадение траекторий"
    )

    plt.xlabel("Δx, км")
    plt.ylabel("Δy, км")

    plt.title(
        f"Фазовая кривая отклонения: {params.name}"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

def compare_perturbed_unperturbed(
    tau_values: np.ndarray,
    q_perturbed: np.ndarray,
    q_unperturbed: np.ndarray,
    params: SystemParams
):
    x_pert = q_perturbed[:, 0]
    y_pert = q_perturbed[:, 1]

    x_unpert = q_unperturbed[:, 0]
    y_unpert = q_unperturbed[:, 1]

    dx = x_pert - x_unpert
    dy = y_pert - y_unpert

    dr = np.sqrt(dx**2 + dy**2)

    max_dx = np.max(np.abs(dx))
    max_dy = np.max(np.abs(dy))
    max_dr = np.max(dr)
    max_dr_km = max_dr * params.a2_km

    print("\nСравнение возмущенного и невозмущенного движения:")
    print(f"max |Δx|  = {max_dx:.10e}")
    print(f"max |Δy|  = {max_dy:.10e}")
    print(f"max |Δr|  = {max_dr:.10e}")
    print(f"max ΔR км = {max_dr_km:.6f} км")

    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, dx, label="Δx(τ)")
    plt.plot(tau_values, dy, label="Δy(τ)")
    plt.xlabel("τ")
    plt.ylabel("Отклонение координат")
    plt.title(f"Разность координат: {params.name}")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, dr)
    plt.xlabel("τ")
    plt.ylabel("|Δr|")
    plt.title(f"Модуль отклонения: {params.name}")
    plt.grid(True)
    plt.show()


def plot_trajectory_comparison(
    q_perturbed: np.ndarray,
    q_unperturbed: np.ndarray,
    params: SystemParams
):
    # Безразмерные координаты
    x_pert = q_perturbed[:, 0]
    y_pert = q_perturbed[:, 1]
    x_unpert = q_unperturbed[:, 0]
    y_unpert = q_unperturbed[:, 1]

    # Перевод в километры
    x_pert_km = x_pert * params.a2_km
    y_pert_km = y_pert * params.a2_km

    x_unpert_km = x_unpert * params.a2_km
    y_unpert_km = y_unpert * params.a2_km

    dx = (x_pert - x_unpert) * params.a2_km
    dy = (y_pert - y_unpert) * params.a2_km
    dr = np.sqrt(dx**2 + dy**2)
    steps = np.arange(len(dr))
    tau   = steps / (len(steps) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Левый: полная орбита (для контекста) ---
    ax1 = axes[0]
    ax1.plot(
        x_pert_km,
        y_pert_km,
        color="tab:blue",
        label="С учётом возмущения",
        linewidth=1.5
    )

    ax1.plot(
        x_unpert_km,
        y_unpert_km,
        "--",
        color="tab:orange",
        label="Без учёта возмущения",
        linewidth=1.5
    )
    ax1.scatter([0], [0], marker="o", color="black", zorder=5,
                label=params.planet)
    ax1.set_xlabel("x, км")
    ax1.set_ylabel("y, км")
    ax1.set_title(f"Орбита: {params.name}")
    ax1.axis("equal")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    # --- Правый: модуль отклонения |Δr(τ)| в километрах ---
    ax2 = axes[1]
    ax2.plot(tau, dr, color="purple", linewidth=2)
    ax2.fill_between(tau, dr, alpha=0.15, color="purple")

    # Точка максимума
    idx_max = np.argmax(dr)
    ax2.scatter(tau[idx_max], dr[idx_max],
                color="red", zorder=5, s=60)
    ax2.annotate(
        f"max Δr = {dr[idx_max]:.0f} км",
        xy=(tau[idx_max], dr[idx_max]),
        xytext=(tau[idx_max] - 0.25, dr[idx_max] * 0.75),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        fontsize=10, color="red"
    )

    ax2.set_xlabel("τ (безразмерное время, 1 = 1 период)")
    ax2.set_ylabel("|Δr|, км")
    ax2.set_title("Отклонение возмущённой траектории от невозмущённой")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.grid(True)

    plt.suptitle(f"Сравнение траекторий: {params.name}", fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_results(tau_values: np.ndarray, q_values: np.ndarray, params: SystemParams):

    x = q_values[:, 0]
    y = q_values[:, 1]

    # --- 1. Безразмерный график ---
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label=f"{params.satellite}")
    plt.scatter([0], [0], marker="o", label=params.planet)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Безразмерная траектория: {params.name}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- 2. В километрах ---
    x_km = x * params.a2_km
    y_km = y * params.a2_km

    plt.figure(figsize=(6, 6))
    plt.plot(x_km, y_km, label=f"{params.satellite}")
    plt.scatter([0], [0], marker="o", label=params.planet)
    plt.xlabel("x, км")
    plt.ylabel("y, км")
    plt.title(f"Траектория в км: {params.name}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- 3. Зависимости от времени ---
    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, x, label="x(τ)")
    plt.plot(tau_values, y, label="y(τ)")
    plt.xlabel("τ")
    plt.ylabel("Координаты (безразмерные)")
    plt.title(f"Координаты от времени: {params.name}")
    plt.grid(True)
    plt.legend()
    plt.show()

@property
def xi(self) -> float:
    r23 = self.a2_km
    r12 = self.a1_au * AU_KM
    return self.m1_over_m2 * (r23 / r12) ** 3

def analyze_all_systems(tau_max: float = 1.0, h: float = 0.001):
    systems = build_system_catalog()

    xi_values = []
    deviation_values = []
    names = []

    for params in systems:
        tau, q_p = integrate(
            params,
            tau_max=tau_max,
            h=h,
            perturbation=True
        )

        _, q_u = integrate(
            params,
            tau_max=tau_max,
            h=h,
            perturbation=False
        )

        dx = q_p[:, 0] - q_u[:, 0]
        dy = q_p[:, 1] - q_u[:, 1]

        dr = np.sqrt(dx**2 + dy**2)

        xi_values.append(params.xi)
        deviation_values.append(np.max(dr) * params.a2_km)
        names.append(params.name)

    return xi_values, deviation_values, names

def plot_xi_vs_deviation(
    xi_values,
    deviation_values,
    names
):
    plt.figure(figsize=(10, 6))

    plt.scatter(xi_values, deviation_values)

    for i, name in enumerate(names):
        plt.annotate(
            name,
            (xi_values[i], deviation_values[i]),
            fontsize=8
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Параметр возмущения ξ")
    plt.ylabel("Максимальное отклонение ΔR, км")

    plt.title(
        "Зависимость отклонения траектории от параметра ξ"
    )

    plt.grid(True, which="both")

    plt.show()

def animate_orbit(
    q_values: np.ndarray,
    params: SystemParams,
    interval: int = 20,
    step: int = 3
):
    x = q_values[:, 0]
    y = q_values[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(x, y, linewidth=1, label="Траектория")
    ax.scatter([0], [0], marker="o", s=80, label=params.planet)

    satellite_point, = ax.plot([], [], "o", markersize=8, label=params.satellite)
    trace_line, = ax.plot([], [], linewidth=2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Анимация движения спутника: {params.name}")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    def init():
        satellite_point.set_data([], [])
        trace_line.set_data([], [])
        return satellite_point, trace_line

    def update(frame):
        i = frame * step

        if i >= len(x):
            i = len(x) - 1

        satellite_point.set_data([x[i]], [y[i]])
        trace_line.set_data(x[:i + 1], y[:i + 1])

        return satellite_point, trace_line

    frames = len(x) // step

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=True
    )

    plt.show()

    return anim

def plot_radius_vector(tau_values, q_perturbed, q_unperturbed, params):
    r_p = np.sqrt(q_perturbed[:, 0]**2 + q_perturbed[:, 1]**2)
    r_u = np.sqrt(q_unperturbed[:, 0]**2 + q_unperturbed[:, 1]**2)

    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, r_p, label="С учетом Солнца")
    plt.plot(tau_values, r_u, "--", label="Без учета Солнца")

    plt.xlabel("τ")
    plt.ylabel("r(τ)")
    plt.title(f"Изменение модуля радиус-вектора: {params.name}")
    plt.grid(True)
    plt.legend()
    plt.show()

def save_results(filename: str, tau_values: np.ndarray, q_values: np.ndarray):
    data = np.column_stack([tau_values, q_values])
    header = "tau,x,y,u,v,theta1"
    np.savetxt(filename, data, delimiter=",", header=header, comments="")


def main():
    params = choose_system()

    print("\nВыбранная система:")
    print(f"Планета: {params.planet}")
    print(f"Спутник: {params.satellite}")
    print(f"m1/m2   = {params.m1_over_m2}")
    print(f"eta     = {params.eta}")
    print(f"a1      = {params.a1_au} а.е.")
    print(f"a2      = {params.a2_km} км")
    print(f"e1      = {params.e1}")
    print(f"e2      = {params.e2}")
    print(f"mu      = {params.mu}")
    print(f"lambda  = {params.lambda_}")
    print(f"nu      = {params.nu}")

    tau_max = 1.0
    h = 0.001

    tau_values, q_perturbed = integrate(
        params,
        tau_max=tau_max,
        h=h,
        perturbation=True
    )

    _, q_unperturbed = integrate(
        params,
        tau_max=tau_max,
        h=h,
        perturbation=False
    )
    print("[x_0,y_0,u,v,theta_1]: ",satellite_initial_conditions(params))
    print(f"xi      = {params.xi:.10e}")
    print(f"xi+eta  = {params.xi + params.eta:.10e}")
    filename = f"orbit_results_{params.planet}_{params.satellite}.csv"
    filename = filename.replace(" ", "_").replace("—", "-")

    save_results(filename, tau_values, q_perturbed)
    plot_results(tau_values, q_perturbed, params)
    animate_orbit(
        q_perturbed,
        params
    )
    plot_trajectory_comparison(
        q_perturbed,
        q_unperturbed,
        params
    )

    plot_radius_vector(
        tau_values,
        q_perturbed,
        q_unperturbed,
        params
    )

    plot_deviation_trajectory(
        q_perturbed,
        q_unperturbed,
        params
    )

    compare_perturbed_unperturbed(
        tau_values,
        q_perturbed,
        q_unperturbed,
        params
    )

    xi_values, deviation_values, names = analyze_all_systems()

    plot_xi_vs_deviation(
        xi_values,
        deviation_values,
        names
    )
    print("\nРасчет завершен.")
    print(f"Результаты сохранены в файл: {filename}")



if __name__ == "__main__":
    main()
