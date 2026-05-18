import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")     # Backend для tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os

#  КОНСТАНТИ ДИЗАЙНУ
BG_DARK      = "#f5f3ee"   # фон головного вікна
BG_CARD      = "#fffdf8"   # фон карток
BG_PANEL     = "#eef4ea"   # фон бічної панелі
ACCENT       = "#ff7a00"   # акцентний оранжевий
ACCENT2      = "#4c956c"   # зелений акцент
TEXT_WHITE   = "#2b2b2b"   # основний текст
TEXT_MUTED   = "#7a7a7a"   # приглушений текст
GREEN        = "#52b788"   # зелений — низький ризик
YELLOW       = "#f4a261"   # жовтий — середній ризик
RED          = "#e76f51"   # червоний — високий ризик

FONT_H1  = ("Segoe UI", 16, "bold")
FONT_H2  = ("Segoe UI", 12, "bold")
FONT_BODY= ("Segoe UI", 10)
FONT_SM  = ("Segoe UI", 9)


#  ШІ-МОДУЛЬ: Лінійна регресія методом найменших квадратів (OLS) + класифікатор

class LinearRegressionOLS:
    def __init__(self):
        self.weights = None   # [w0 (intercept), w1 (slope)]

    # Навчання моделі на основі даних (X: тижні, y: витрати)
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Додаємо стовпець одиниць для вільного члена (bias)
        X_b = np.column_stack([np.ones(len(X)), X])   # shape (n, 2)
        # Аналітичне рішення OLS
        # (X^T X)^{-1} X^T y
        XtX = X_b.T @ X_b                              
        Xty = X_b.T @ y                               
        self.weights = np.linalg.pinv(XtX) @ Xty

    # Прогноз для одного значення x
    def predict(self, x: float) -> float:
        if self.weights is None:
            raise RuntimeError("Модель не навчена. Викличте fit() спочатку.")
        return self.weights[0] + self.weights[1] * x   # w0 + w1*x

    # Прогноз для масиву значень x
    def predict_series(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x) for x in X])


# Класифікатор ризику перевищення бюджету
class RiskClassifier:

    def classify(self, expenses: dict, budget: float) -> dict:
        if budget <= 0:
            return {"score": 0, "level": "—", "color": TEXT_MUTED, "advice": "Встановіть бюджет."}

        total = sum(expenses.values())
        if total == 0:
            return {"score": 0, "level": "🟢 Низький", "color": GREEN,
                    "advice": "Витрат не введено."}

        # Базовий ризик: скільки % бюджету витрачено за тиждень
        base_ratio = total / budget

        # Структурний множник: чим більша частка покупок — тим вищий ризик
        shop_ratio = expenses.get("shopping", 0) / total
        multiplier = 1.0 + 0.4 * shop_ratio   # від 1.0 до 1.4

        # Фінальний ризик (%)
        score = base_ratio * 100 * multiplier

        # Класифікація
        if score < 40:
            level  = "🟢 Низький"
            color  = GREEN
            advice = "Відмінно! Витрати в межах норми. Продовжуйте в тому ж дусі."
        elif score < 80:
            level  = "🟡 Середній"
            color  = YELLOW
            advice = "Увага! Витрати наближаються до ліміту. Перегляньте покупки."
        else:
            level  = "🔴 Високий"
            color  = RED
            advice = "⚠ Ризик перевищення бюджету! Скоротіть витрати негайно."

        return {
            "score":  round(score, 1),
            "level":  level,
            "color":  color,
            "advice": advice,
        }


#  ГОЛОВНИЙ КЛАС ДОДАТКУ
# ─────────────────────────────────────────────
class SmartExpenseApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart Expense Analyzer - AI аналіз витрат")
        self.root.geometry("1050x720")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # ШІ-компоненти
        self.regression = LinearRegressionOLS()
        self.classifier  = RiskClassifier()

        # Дані — список тижневих записів: [{food, transport, shopping}, ...]
        self.history: list[dict] = []
        self.data_file = "expense_history.json"
        self._load_history()

        # Побудова UI
        self._build_menu()
        self._build_ui()

    # ──────────────── МЕНЮ ────────────────
    def _build_menu(self):
        menubar = tk.Menu(self.root, bg=BG_PANEL, fg=TEXT_WHITE,
                          activebackground=ACCENT, activeforeground=TEXT_WHITE,
                          relief="flat")
        self.root.config(menu=menubar)

        # Файл
        file_menu = tk.Menu(menubar, tearoff=0, bg=BG_PANEL, fg=TEXT_WHITE,
                            activebackground=ACCENT)
        menubar.add_cascade(label="  Файл  ", menu=file_menu)
        file_menu.add_command(label="Очистити історію", command=self._clear_history)
        file_menu.add_separator()
        file_menu.add_command(label="Вийти", command=self.root.quit)

        # Довідка
        help_menu = tk.Menu(menubar, tearoff=0, bg=BG_PANEL, fg=TEXT_WHITE,
                            activebackground=ACCENT)
        menubar.add_cascade(label="  Довідка  ", menu=help_menu)
        help_menu.add_command(label="Інструкція користувача", command=self._show_help)
        help_menu.add_command(label="Про програму",           command=self._show_about)

    # ──────────────── ГОЛОВНИЙ UI ────────────────
    def _build_ui(self):
        # ── Заголовок
        header = tk.Frame(self.root, bg=ACCENT2, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="💰 Smart Expense Analyzer",
                 font=("Segoe UI", 15, "bold"),
                 bg=ACCENT2, fg=TEXT_WHITE).pack(side="left", padx=20, pady=12)
        tk.Label(header, text="© Yurchyshyn Solomia 2026 | AI аналіз витрат",
                 font=FONT_SM, bg=ACCENT2, fg=TEXT_WHITE).pack(side="right", padx=20)

        # ── Основна область (ліво + право)
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Ліва панель — введення
        self._build_input_panel(main)

        # Права панель — результати + графік
        self._build_result_panel(main)

    def _build_input_panel(self, parent):
        left = tk.Frame(parent, bg=BG_PANEL, width=320)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        # Заголовок
        tk.Label(left, text="Введення даних", font=FONT_H2,
                 bg=BG_PANEL, fg=ACCENT).pack(pady=(15, 5), padx=15, anchor="w")
        tk.Label(left, text="Введіть витрати за поточний тиждень (грн)",
                 font=FONT_SM, bg=BG_PANEL, fg=TEXT_MUTED,
                 wraplength=280, justify="left").pack(padx=15, anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=15, pady=10)

        # Поля категорій
        categories = [
            ("🍕  Їжа (грн)", "food"),
            ("🚌  Транспорт (грн)", "transport"),
            ("🛍️  Покупки/розваги (грн)", "shopping"),
        ]
        self.entries: dict[str, tk.StringVar] = {}

        for label_text, key in categories:
            frame = tk.Frame(left, bg=BG_PANEL)
            frame.pack(fill="x", padx=15, pady=6)

            tk.Label(frame, text=label_text, font=FONT_BODY,
                     bg=BG_PANEL, fg=TEXT_WHITE).pack(anchor="w")

            var = tk.StringVar(value="0")
            entry = tk.Entry(frame, textvariable=var, font=("Segoe UI", 13, "bold"),
                             bg=BG_CARD, fg=ACCENT, insertbackground=ACCENT,
                             relief="flat", bd=5, justify="right", width=15)
            entry.pack(fill="x", ipady=6)
            self.entries[key] = var

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=15, pady=10)

        # Бюджет
        tk.Label(left, text="💳  Місячний бюджет (грн)", font=FONT_BODY,
                 bg=BG_PANEL, fg=TEXT_WHITE).pack(padx=15, anchor="w")
        self.budget_var = tk.StringVar(value="10000")
        tk.Entry(left, textvariable=self.budget_var,
                 font=("Segoe UI", 13, "bold"), bg=BG_CARD, fg=GREEN,
                 insertbackground=GREEN, relief="flat", bd=5,
                 justify="right", width=15).pack(fill="x", padx=15, ipady=6, pady=(3, 12))

        # Кнопка аналізу
        tk.Button(left, text="🔍  Аналізувати",
                  font=("Segoe UI", 12, "bold"),  bg=ACCENT, fg="white",
                  activebackground="#c0392b", activeforeground="white",
                  relief="flat", cursor="hand2", bd=0,  command=self._run_analysis,
                  pady=10).pack(fill="x", padx=15, pady=(0, 8))

        # Кнопка додати в історію
        tk.Button(left, text="➕  Додати тиждень до історії",
                  font=FONT_BODY,  bg=ACCENT2, fg=TEXT_WHITE,
                  activebackground="#0a2d5e", activeforeground=TEXT_WHITE,
                  relief="flat", cursor="hand2", bd=0,  command=self._add_to_history,
                  pady=7).pack(fill="x", padx=15, pady=(0, 5))

        # Лічильник записів
        self.history_label = tk.Label(left, text=f"Записів в історії: {len(self.history)}",
                                      font=FONT_SM, bg=BG_PANEL, fg=TEXT_MUTED)
        self.history_label.pack(padx=15, anchor="w")

        # Підказки
        tk.Label(left, text="\nℹ️Меню → Довідка → Інструкція",
                 font=FONT_SM, bg=BG_PANEL, fg=TEXT_MUTED,
                 justify="left").pack(padx=15, anchor="w")

    # ──────────────── ПРАВА ПАНЕЛЬ РЕЗУЛЬТАТІВ ────────────────
    def _build_result_panel(self, parent):
        right = tk.Frame(parent, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)

        # ── Карточки результатів
        cards_row = tk.Frame(right, bg=BG_DARK)
        cards_row.pack(fill="x", pady=(0, 8))

        # Картка: поточні витрати
        self.total_card = self._make_card(cards_row, "Поточні витрати", "0 грн", TEXT_WHITE)
        self.total_card.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Картка: прогноз
        self.forecast_card = self._make_card(cards_row, "Прогноз (наст. тиждень)", "— грн", TEXT_MUTED)
        self.forecast_card.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Картка: ризик
        self.risk_card = self._make_card(cards_row, "Ризик перевищення", "-", TEXT_MUTED)
        self.risk_card.pack(side="left", fill="both", expand=True)

        # ── Порада AI
        advice_frame = tk.Frame(right, bg=BG_CARD, pady=8)
        advice_frame.pack(fill="x", pady=(0, 8))
        tk.Label(advice_frame, text="💡 Порада AI:", font=FONT_H2,
                 bg=BG_CARD, fg=ACCENT).pack(side="left", padx=12)
        self.advice_label = tk.Label(advice_frame,
                                     text="Введіть дані та натисніть «Аналізувати»",
                                     font=FONT_BODY, bg=BG_CARD, fg=TEXT_MUTED, wraplength=550, justify="left")
        self.advice_label.pack(side="left", padx=5)

        # ── Прогрес-бар ризику
        bar_frame = tk.Frame(right, bg=BG_DARK)
        bar_frame.pack(fill="x", pady=(0, 8))
        tk.Label(bar_frame, text="Оцінка ризику:", font=FONT_SM,
                 bg=BG_DARK, fg=TEXT_MUTED).pack(side="left", padx=(0, 8))
        self.risk_bar = ttk.Progressbar(bar_frame, length=400, mode="determinate",
                                        maximum=100)
        self.risk_bar.pack(side="left", fill="x", expand=True)
        self.risk_pct_label = tk.Label(bar_frame, text="0%", font=FONT_SM,
                                       bg=BG_DARK, fg=TEXT_MUTED, width=6)
        self.risk_pct_label.pack(side="left", padx=8)

        # ── Графік matplotlib
        graph_frame = tk.Frame(right, bg=BG_CARD)
        graph_frame.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(6, 3.5), facecolor=BG_CARD)
        self.ax  = self.fig.add_subplot(111)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty_chart()

    def _make_card(self, parent, title: str, value: str, color: str) -> tk.Frame:
        card = tk.Frame(parent, bg=BG_CARD, pady=10, padx=12)
        lbl_title = tk.Label(card, text=title, font=FONT_SM,
                              bg=BG_CARD, fg=TEXT_MUTED)
        lbl_title.pack(anchor="w")
        lbl_val = tk.Label(card, text=value, font=("Segoe UI", 18, "bold"),
                           bg=BG_CARD, fg=color)
        lbl_val.pack(anchor="w")
        # Зберігаємо посилання на label значення
        card._val_label = lbl_val
        return card

    # ──────────────── АНАЛІЗ ────────────────
    def _get_expenses(self) -> dict | None:
        try:
            expenses = {
                "food": float(self.entries["food"].get()),
                "transport": float(self.entries["transport"].get()),
                "shopping": float(self.entries["shopping"].get()),
            }
            budget = float(self.budget_var.get())
            if any(v < 0 for v in expenses.values()) or budget <= 0:
                raise ValueError
            return expenses, budget
        except ValueError:
            messagebox.showerror(
                "Помилка введення",
                "Перевірте поля! Всі значення мають бути числами ≥ 0.\n"
                "Бюджет має бути > 0.",
            )
            return None, None

    # Основна функція аналізу — викликається кнопкою
    def _run_analysis(self):
        expenses, budget = self._get_expenses()
        if expenses is None:
            return

        total = sum(expenses.values())

        # ── 1. Класифікація ризику
        risk = self.classifier.classify(expenses, budget)

        # ── 2. Лінійна регресія (потребує ≥ 2 тижні в історії)
        forecast_text = "Потрібно ≥ 2 тижні"
        trend_slope   = None
        if len(self.history) >= 2:
            weeks  = np.array([1.0 + i for i in range(len(self.history))])
            totals = np.array([sum(w.values()) for w in self.history], dtype=float)
            self.regression.fit(weeks, totals)
            next_week = len(self.history) + 1.0
            forecast  = self.regression.predict(next_week)
            forecast_text = f"{forecast:,.0f} грн"
            trend_slope = self.regression.weights[1]  # нахил тренду

        # ── 3. Оновлення карток
        self.total_card._val_label.config(text=f"{total:,.0f} грн", fg=TEXT_WHITE)
        self.forecast_card._val_label.config(text=forecast_text,
                                             fg=YELLOW if forecast_text != "Потрібно ≥ 2 тижні" else TEXT_MUTED)
        self.risk_card._val_label.config(text=risk["level"], fg=risk["color"])

        # ── 4. Порада AI (з урахуванням тренду)
        advice = risk["advice"]
        if trend_slope is not None:
            if trend_slope > 50:
                advice += f"  📈 Тренд: витрати зростають (+{trend_slope:.0f} грн/тиждень)."
            elif trend_slope < -50:
                advice += f"  📉 Тренд: витрати спадають ({trend_slope:.0f} грн/тиждень). Так тримати!"
        self.advice_label.config(text=advice, fg=risk["color"])

        # ── 5. Прогрес-бар
        bar_val = min(risk["score"], 100)
        self.risk_bar["value"] = bar_val
        self.risk_pct_label.config(text=f'{risk["score"]}%', fg=risk["color"])

        # ── 6. Графік
        self._update_chart(expenses, budget)

    # Додає поточні витрати в історію тижнів
    def _add_to_history(self):
        expenses, budget = self._get_expenses()
        if expenses is None:
            return
        self.history.append(expenses)
        self._save_history()
        self.history_label.config(text=f"Записів в історії: {len(self.history)}")
        messagebox.showinfo("Готово", f"Тиждень #{len(self.history)} збережено!")
        # Автоматично оновити аналіз
        self._run_analysis()

    # ──────────────── ГРАФІК ────────────────
    def _style_axes(self):
        self.ax.set_facecolor(BG_CARD)
        self.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_MUTED)
        self.ax.yaxis.label.set_color(TEXT_MUTED)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(ACCENT2)

    def _draw_empty_chart(self):
        self.ax.clear()
        self._style_axes()
        self.ax.set_title("Структура витрат та прогноз", color=TEXT_WHITE, pad=8)
        self.ax.text(0.5, 0.5, "Введіть дані для відображення",
                     transform=self.ax.transAxes, ha="center", va="center",
                     color=TEXT_MUTED, fontsize=10)
        self.canvas.draw()

    def _update_chart(self, expenses: dict, budget: float):
        self.ax.clear()
        self._style_axes()
        self.fig.clear()

        # ── Якщо є ≥ 2 тижні — показуємо лінійний графік з прогнозом
        if len(self.history) >= 2:
            ax1 = self.fig.add_subplot(121, facecolor=BG_CARD)
            ax2 = self.fig.add_subplot(122, facecolor=BG_CARD)
            self.fig.patch.set_facecolor(BG_CARD)

            # Графік 1: Тижневі витрати + лінія регресії + прогноз
            weeks  = np.arange(1, len(self.history) + 1, dtype=float)
            totals = np.array([sum(w.values()) for w in self.history])

            # Лінія регресії
            x_reg = np.linspace(1, len(self.history) + 1, 50)
            y_reg = self.regression.predict_series(x_reg)

            ax1.plot(weeks, totals, "o-", color=ACCENT, linewidth=2,
                     markersize=6, label="Фактичні")
            ax1.plot(x_reg, y_reg, "--", color=YELLOW, linewidth=1.5,
                     label="Регресія")
            # Прогнозна точка
            forecast = self.regression.predict(len(self.history) + 1)
            ax1.plot(len(self.history) + 1, forecast, "^",
                     color=GREEN, markersize=10, label=f"Прогноз: {forecast:.0f}")
            # Лінія бюджету
            ax1.axhline(budget, color=RED, linestyle=":", linewidth=1.2,
                        label=f"Бюджет: {budget:.0f}")

            ax1.set_title("Тренд витрат", color=TEXT_WHITE, pad=6)
            ax1.set_xlabel("Тиждень", color=TEXT_MUTED)
            ax1.set_ylabel("Грн", color=TEXT_MUTED)
            ax1.tick_params(colors=TEXT_MUTED, labelsize=7)
            for sp in ax1.spines.values(): sp.set_edgecolor(ACCENT2)
            ax1.legend(fontsize=7, facecolor=BG_DARK, labelcolor=TEXT_WHITE,
                       edgecolor=ACCENT2)

            # Графік 2: кругова діаграма поточних витрат
            self._draw_pie(ax2, expenses)
        else:
            # Тільки кругова діаграма
            ax = self.fig.add_subplot(111, facecolor=BG_CARD)
            self.fig.patch.set_facecolor(BG_CARD)
            self._draw_pie(ax, expenses)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

    # Кругова діаграма структури витрат
    def _draw_pie(self, ax, expenses: dict):
        labels = ["Їжа", "Транспорт", "Покупки"]
        values = [expenses["food"], expenses["transport"], expenses["shopping"]]
        colors = [ACCENT, "#fcf951", "#3498db"]
        # Прибрати нульові значення
        filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
        if not filtered:
            ax.text(0.5, 0.5, "Всі витрати = 0",
                    transform=ax.transAxes, ha="center", va="center",
                    color=TEXT_MUTED, fontsize=10)
            return
        fl, fv, fc = zip(*filtered)
        wedges, texts, autotexts = ax.pie(
            fv, labels=fl, colors=fc,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": TEXT_WHITE, "fontsize": 8},
        )
        for at in autotexts:
            at.set_color(BG_DARK)
            at.set_fontsize(8)
        ax.set_title("Структура витрат", color=TEXT_WHITE, pad=6)

    # ──────────────── ЗБЕРЕЖЕННЯ ДАНИХ ────────────────
    def _save_history(self):
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def _load_history(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r", encoding="utf-8") as f:
                self.history = json.load(f)

    def _clear_history(self):
        if messagebox.askyesno("Очистити?", "Видалити всі збережені тижні?"):
            self.history = []
            self._save_history()
            self.history_label.config(text="Записів в історії: 0")
            self._draw_empty_chart()

    # ──────────────── ВІКНА ДОВІДКИ ────────────────
    def _show_help(self):
        win = tk.Toplevel(self.root)
        win.title("Інструкція користувача")
        win.geometry("520x480")
        win.configure(bg=BG_DARK)

        text = (
            "════════════════════════════════\n"
            "   ІНСТРУКЦІЯ КОРИСТУВАЧА\n"
            "   Smart Expense Analyzer\n"
            "════════════════════════════════\n\n"
            "КРОК 1. Введіть витрати за тиждень:\n"
            "  • Їжа (продукти, кафе, ресторани)\n"
            "  • Транспорт (проїзд, пальне, таксі)\n"
            "  • Покупки (одяг, техніка, розваги)\n\n"
            "КРОК 2. Встановіть місячний бюджет (грн).\n\n"
            "КРОК 3. Натисніть «Аналізувати»:\n"
            "  → AI визначить ризик перевищення бюджету\n"
            "  → Отримаєте пораду щодо витрат\n\n"
            "КРОК 4. Натисніть «Додати тиждень до історії»\n"
            "  → Система накопичує дані для прогнозу\n"
            "  → Після 2+ тижнів з'явиться лінія регресії\n"
            "    та прогноз на наступний тиждень\n\n"
            "РІВНІ РИЗИКУ:\n"
            "  🟢 Низький - витрати в нормі\n"
            "  🟡 Середній - наближаємось до ліміту\n"
            "  🔴 Високий - перевищення бюджету!\n\n"
            "Файл → Очистити історію - скидає всі тижні"
        )

        txt = tk.Text(win, bg=BG_CARD, fg=TEXT_WHITE,
                      font=FONT_BODY, relief="flat",
                      padx=20, pady=15, wrap="word")
        txt.insert("1.0", text)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True, padx=10, pady=10)

    def _show_about(self):
        messagebox.showinfo(
            "Про програму",
            "Smart Expense Analyzer v1.0\n\n"
            "Система штучного інтелекту для аналізу\n"
            "особистих витрат і прогнозування бюджету.\n\n"
            "Алгоритми:\n"
            "  • Лінійна регресія (OLS, NumPy)\n"
            "  • Класифікація ризику (зважена сума)\n\n"
            "© Yurchyshyn Solomia 2026\n"
            "Самостійна робота СШІ"
        )


if __name__ == "__main__":
    root = tk.Tk()
    # Налаштування теми ttk (прогрес-бар)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TProgressbar",
                    troughcolor=BG_CARD,
                    background=ACCENT,
                    thickness=12)

    app = SmartExpenseApp(root)
    root.mainloop()