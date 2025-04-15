import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from stock_learn import predict_stock_price

# GUI
class StockApp:
    def __init__(self, master):
        self.master = master
        master.title("Stock Price Predictor")

        self.label = tk.Label(master, text="Enter Stock Ticker (e.g. (BTC-USD):")
        self.label.pack()

        self.entry = tk.Entry(master, width=20)
        self.entry.pack()

        self.predict_button = tk.Button(master, text="Predict", command=self.run_prediction)
        self.predict_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack()

    def run_prediction(self):
        ticker = self.entry.get().strip()
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        try:
            fig, predicted_price = predict_stock_price(ticker)
            self.result_label.config(text=f"Predicted next day's price: ${predicted_price:.2f}")
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
        except Exception as e:
            messagebox.showerror("Error", str(e))


