import customtkinter as ctk
import numpy as np
import pandas as pd
from tkinter import filedialog, messagebox
from datetime import datetime
import threading
import sys
import os

try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    Pool = None

def process_chunk(chunk):
    """Αυτή η συνάρτηση τρέχει ταυτόχρονα σε κάθε πυρήνα της CPU"""
    def calc_expiry(row):
        date_val = row['Ημ. Δημιουργίας Συμβολαίου']
        if pd.isna(date_val): return pd.NaT
        paketo = str(row.get('Πακέτο', '')).lower()
        months = 18 if "max" in paketo else 12
        return date_val + pd.DateOffset(months=months)

    chunk['Ημ. Δημιουργίας Συμβολαίου'] = pd.to_datetime(chunk['Ημ. Δημιουργίας Συμβολαίου'], dayfirst=True, errors='coerce')
    chunk['expiry_dt'] = chunk.apply(calc_expiry, axis=1)
    chunk['calculated_expiry'] = chunk['expiry_dt'].dt.strftime('%d/%m/%Y').fillna("-")
    return chunk

if __name__ != "__main__":
    import __main__
    __main__.process_chunk = process_chunk

class MultiSelectWindow(ctk.CTkToplevel):
    def __init__(self, parent, title, options, selected_set, callback, is_package=False):
        super().__init__(parent)
        self.title(title)
        self.geometry("500x750")
        self.attributes("-topmost", True)
        ctk.set_appearance_mode("dark")  # Επιλογές: "dark", "light", "system"
        ctk.set_default_color_theme("dark-blue")  # Επιλογές: "blue", "green", "dark-blue"
        self.selected_set = selected_set
        self.callback = callback
        self.checkboxes = []
        ctrl_frame = ctk.CTkFrame(self)
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        top_btn_f = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        top_btn_f.pack(fill="x", pady=5)
        ctk.CTkButton(top_btn_f, text="✅ Όλα", command=self.select_all, width=100).pack(side="left", padx=5, expand=True)
        ctk.CTkButton(top_btn_f, text="❌ Καθαρισμός", command=self.deselect_all, fg_color="firebrick", width=100).pack(side="left", padx=5, expand=True)
        if is_package:
            smart_f = ctk.CTkFrame(ctrl_frame)
            smart_f.pack(fill="x", padx=5, pady=5)
            ctk.CTkLabel(smart_f, text="Προσθήκη (+):", font=("Arial", 11, "bold")).grid(row=0, column=0, padx=5, pady=2)
            tags = [("BLUE", "#3b8ed0", "blue"), ("YELLOW", "#f1c40f", "yellow"), ("HOME", "#2ecc71", "home"), ("BUS", "#9b59b6", "business")]
            for i, (txt, clr, kw) in enumerate(tags):
                btn = ctk.CTkButton(smart_f, text=txt, fg_color=clr, text_color="black" if i==1 else "white", width=80, height=24, command=lambda k=kw: self.mass_toggle(k, True))
                btn.grid(row=0, column=i+1, padx=2, pady=2)
            ctk.CTkLabel(smart_f, text="Αφαίρεση (-):", font=("Arial", 11, "bold")).grid(row=1, column=0, padx=5, pady=2)
            for i, (txt, clr, kw) in enumerate(tags):
                btn = ctk.CTkButton(smart_f, text=txt, fg_color="#444", border_width=1, border_color=clr, width=80, height=24, command=lambda k=kw: self.mass_toggle(k, False))
                btn.grid(row=1, column=i+1, padx=2, pady=2)
        self.scroll = ctk.CTkScrollableFrame(self, label_text="Λίστα Πακέτων")
        self.scroll.pack(fill="both", expand=True, padx=10, pady=5)
        for option in options:
            var = ctk.BooleanVar(value=option in self.selected_set)
            cb = ctk.CTkCheckBox(self.scroll, text=str(option), variable=var, command=lambda o=option, v=var: self.toggle_option(o, v))
            cb.pack(anchor="w", pady=4)
            self.checkboxes.append((cb, var))
        ctk.CTkButton(self, text="ΕΦΑΡΜΟΓΗ ΦΙΛΤΡΩΝ", command=self.close_window, fg_color="#27ae60", height=50, font=("Arial", 14, "bold")).pack(pady=10, fill="x", padx=20)
    
    def toggle_option(self, option, var):
        if var.get(): self.selected_set.add(option)
        else: self.selected_set.discard(option)
    def mass_toggle(self, keyword, state):
        for cb, var in self.checkboxes:
            if keyword.lower() in cb.cget("text").lower():
                var.set(state)
                if state: self.selected_set.add(cb.cget("text"))
                else: self.selected_set.discard(cb.cget("text"))
    def select_all(self):
        for cb, var in self.checkboxes:
            var.set(True)
            self.selected_set.add(cb.cget("text"))
    def deselect_all(self):
        for cb, var in self.checkboxes: var.set(False)
        self.selected_set.clear()
    def close_window(self):
        self.callback()
        self.destroy()

class MegaEnergyCRM(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MEGA ENERGY CRM")
        self.geometry("1450x900")
        ctk.CTkLabel(self, text="Version 1.1.1").pack() # Ενημερωμένη έκδοση
        
        self.df = None
        self.filtered_df = pd.DataFrame()
        self.current_page = 0
        self.page_size = 50  
        self.selected_agents = set()
        self.selected_packets = set()
        self.selected_metrites = set()
        self.available_info = {
            "Agent Code": "AgentCode", "ΑΦΜ": "ΑΦΜ", "Κινητό": "Κινητό", 
            "Email": "Email", "Πακέτο": "Πακέτο",
            "Μετρητής": "Κατάσταση Μετρητή", "Παροχή": "Αριθμός Παροχής",
            "Λήξη Συμβολαίου": "calculated_expiry"
        }
        self.show_vars = {}
        self.sidebar = ctk.CTkScrollableFrame(self, width=320, label_text="MEGA ENERGY CONTROL")
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        ctk.CTkLabel(self.sidebar, text="MEGA ENERGY", font=("Arial Black", 26), text_color="#3b8ed0").pack(pady=(15, 5))
        self.btn_load = ctk.CTkButton(self.sidebar, text="📂 1. Φόρτωση Excel", command=self.start_load_thread, height=45)
        self.btn_load.pack(pady=10, padx=10, fill="x")
        self.add_section_label("📅 Ημ. Δημιουργίας")
        self.entry_start = ctk.CTkEntry(self.sidebar, placeholder_text="Από: DD/MM/YYYY")
        self.entry_start.pack(pady=2, padx=10, fill="x")
        self.entry_start.bind("<KeyRelease>", lambda e: self.format_date_input(e, self.entry_start))
        self.entry_end = ctk.CTkEntry(self.sidebar, placeholder_text="Έως: DD/MM/YYYY")
        self.entry_end.pack(pady=2, padx=10, fill="x")
        self.entry_end.bind("<KeyRelease>", lambda e: self.format_date_input(e, self.entry_end))
        self.expiry_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.add_section_label("🔔 Ημ. Λήξης (Blue Only)", master=self.expiry_frame)
        self.expiry_start = ctk.CTkEntry(self.expiry_frame, placeholder_text="Από: DD/MM/YYYY")
        self.expiry_start.pack(pady=2, padx=10, fill="x")
        self.expiry_start.bind("<KeyRelease>", lambda e: self.format_date_input(e, self.expiry_start))
        self.expiry_end = ctk.CTkEntry(self.expiry_frame, placeholder_text="Έως: DD/MM/YYYY")
        self.expiry_end.pack(pady=2, padx=10, fill="x")
        self.expiry_end.bind("<KeyRelease>", lambda e: self.format_date_input(e, self.expiry_end))
        self.btn_apply = ctk.CTkButton(self.sidebar, text="🔍 ΕΦΑΡΜΟΓΗ ΦΙΛΤΡΩΝ", command=self.apply_filters, fg_color="#27ae60", height=45)
        self.btn_apply.pack(pady=15, padx=10, fill="x")
        self.btn_excel = ctk.CTkButton(self.sidebar, text="📊 ΕΞΑΓΩΓΗ ΣΕ EXCEL", command=self.export_to_excel, fg_color="#2980b9", height=40)
        self.btn_excel.pack(pady=5, padx=10, fill="x")
        self.add_section_label("Φίλτρα Επιλογών")
        self.btn_filter_paketo = ctk.CTkButton(self.sidebar, text="📦 Επιλογή Πακέτων", command=lambda: self.open_multiselect("Πακέτο", self.selected_packets, "Επιλογή Πακέτων", True), fg_color="gray30")
        self.btn_filter_paketo.pack(pady=5, padx=10, fill="x")
        self.btn_filter_agent = ctk.CTkButton(self.sidebar, text="👥 Επιλογή Agents", command=lambda: self.open_multiselect("AgentCode", self.selected_agents, "Επιλογή Agents"), fg_color="gray30")
        self.btn_filter_agent.pack(pady=5, padx=10, fill="x")
        self.btn_filter_metritis = ctk.CTkButton(self.sidebar, text="⚡ Κατάσταση Μετρητή", command=lambda: self.open_multiselect("Κατάσταση Μετρητή", self.selected_metrites, "Επιλογή Μετρητή"), fg_color="gray30")
        self.btn_filter_metritis.pack(pady=5, padx=10, fill="x")
        self.add_section_label("Εμφάνιση στην Κάρτα")
        for label, col_name in self.available_info.items():
            var = ctk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(self.sidebar, text=label, variable=var, command=self.render_page)
            cb.pack(pady=3, padx=20, anchor="w")
            self.show_vars[col_name] = var
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        self.search_entry = ctk.CTkEntry(self.main_container, placeholder_text="🔍 Αναζήτηση...", height=50)
        self.search_entry.pack(fill="x", pady=(0, 10))
        self.search_entry.bind("<KeyRelease>", lambda e: self.apply_filters())
        self.results_area = ctk.CTkScrollableFrame(self.main_container, label_text="Λίστα Αποτελεσμάτων")
        self.results_area.pack(fill="both", expand=True)
        self.nav_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.nav_frame.pack(fill="x", pady=10)
        ctk.CTkButton(self.nav_frame, text="◀ Πίσω", command=self.prev_page, width=100).pack(side="left", padx=20)
        self.page_label = ctk.CTkLabel(self.nav_frame, text="Σελίδα 0 από 0", font=("Arial", 13, "bold"))
        self.page_label.pack(side="left", expand=True)
        ctk.CTkButton(self.nav_frame, text="Επόνη ▶", command=self.next_page, width=100).pack(side="right", padx=20)
        self.status_label = ctk.CTkLabel(self.main_container, text="Αναμονή αρχείου...")
        self.status_label.pack(pady=5)

    def format_date_input(self, event, entry_widget):
        if event.keysym == "BackSpace": return
        text = entry_widget.get()
        clean = ''.join(filter(str.isdigit, text))
        formatted = clean
        if len(clean) > 2: formatted = clean[:2] + '/' + clean[2:]
        if len(clean) > 4: formatted = formatted[:5] + '/' + formatted[5:]
        if len(formatted) > 10: formatted = formatted[:10]
        if text != formatted:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, formatted)

    def start_load_thread(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx *.xls *.csv")])
        if path:
            self.btn_load.configure(state="disabled", text="⏳ Επεξεργασία...")
            thread = threading.Thread(target=self.threaded_load, args=(path,))
            thread.daemon = True
            thread.start()

    def threaded_load(self, path):
        try:
            raw_df = pd.read_excel(path) if not path.endswith('.csv') else pd.read_csv(path)
            
            if raw_df.empty:
                self.after(0, lambda: messagebox.showwarning("Προσοχή", "Το αρχείο είναι άδειο!"))
                self.after(0, self.reset_load_button)
                return

            num_cores = os.cpu_count() or 4
            chunks = np.array_split(raw_df, num_cores)
            chunks = [c for c in chunks if not c.empty]

            if Pool and len(raw_df) > 100:
                with Pool(processes=len(chunks)) as pool:
                    processed_chunks = pool.map(process_chunk, chunks)
                final_df = pd.concat(processed_chunks, ignore_index=True)
            else:
                final_df = process_chunk(raw_df)

            self.after(0, lambda: self.finalize_load(final_df))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Σφάλμα Multiprocessing: {str(e)}"))
            self.after(0, self.reset_load_button)

    def finalize_load(self, df):
        self.df = df
        if "Όνομα Πελάτη" in self.df.columns:
            self.df["Επωνυμία / Ονοματεπώνυμο"] = self.df["Όνομα Πελάτη"]
        self.apply_filters()
        self.reset_load_button()
        self.status_label.configure(text=f"✅ Φορτώθηκαν {len(df)} εγγραφές", text_color="#27ae60")

    def reset_load_button(self):
        self.btn_load.configure(state="normal", text="📂 1. Φόρτωση Excel")

    def apply_filters(self):
        if self.df is None: return
        self.current_page = 0
        f = self.df.copy()
        is_blue_active = any("blue" in str(p).lower() or "μπλε" in str(p).lower() for p in self.selected_packets)
        if is_blue_active: self.expiry_frame.pack(after=self.entry_end, fill="x", pady=5)
        else: self.expiry_frame.pack_forget()
        try:
            s = self.entry_start.get().replace("/", "").strip()
            e = self.entry_end.get().replace("/", "").strip()
            if len(s) == 8: f = f[f['Ημ. Δημιουργίας Συμβολαίου'] >= pd.to_datetime(s, format='%d%m%Y')]
            if len(e) == 8: f = f[f['Ημ. Δημιουργίας Συμβολαίου'] <= pd.to_datetime(e, format='%d%m%Y')]
            if is_blue_active:
                es = self.expiry_start.get().replace("/", "").strip()
                ee = self.expiry_end.get().replace("/", "").strip()
                if len(es) == 8: f = f[f['expiry_dt'] >= pd.to_datetime(es, format='%d%m%Y')]
                if len(ee) == 8: f = f[f['expiry_dt'] <= pd.to_datetime(ee, format='%d%m%Y')]
        except: pass
        if self.selected_packets: f = f[f['Πακέτο'].astype(str).isin(self.selected_packets)]
        if self.selected_agents: f = f[f['AgentCode'].astype(str).isin(self.selected_agents)]
        if self.selected_metrites: f = f[f['Κατάσταση Μετρητή'].astype(str).isin(self.selected_metrites)]
        q = self.search_entry.get().strip().lower()
        if q: f = f[f.astype(str).apply(lambda x: x.str.contains(q, case=False, na=False)).any(axis=1)]
        self.filtered_df = f
        self.update_sidebar_ui()
        self.render_page()

    def update_sidebar_ui(self):
        self.btn_filter_paketo.configure(text=f"📦 Πακέτα ({len(self.selected_packets) or 'Όλα'})", fg_color="#2980b9" if self.selected_packets else "gray30")
        self.btn_filter_agent.configure(text=f"👥 Agents ({len(self.selected_agents) or 'Όλα'})", fg_color="#2980b9" if self.selected_agents else "gray30")
        self.btn_filter_metritis.configure(text=f"⚡ Μετρητές ({len(self.selected_metrites) or 'Όλα'})", fg_color="#2980b9" if self.selected_metrites else "gray30")

    def create_card(self, row):
        paketo = str(row.get('Πακέτο', '')).lower()
        is_blue = "blue" in paketo or "μπλε" in paketo
        is_max = "max" in paketo
        card = ctk.CTkFrame(self.results_area, corner_radius=10, border_width=2 if is_blue else 1, border_color="#3b8ed0" if is_blue else "gray30")
        card.pack(fill="x", padx=15, pady=5)
        name = row.get('Όνομα Πελάτη', row.get('Επωνυμία / Ονοματεπώνυμο', 'N/A'))
        ctk.CTkLabel(card, text=f"{'💎 ' if is_max else ('🔵 ' if is_blue else '')}{name}", font=("Arial", 14, "bold"), text_color="#3b8ed0" if is_blue else "white").pack(anchor="w", padx=15, pady=5)
        info_f = ctk.CTkFrame(card, fg_color="transparent")
        info_f.pack(fill="x", padx=15, pady=5)
        r, c = 0, 0
        for lbl, col in self.available_info.items():
            if lbl == "Λήξη Συμβολαίου" and not is_blue: continue
            if self.show_vars.get(col) and self.show_vars[col].get():
                val = str(row.get(col, "-"))
                l = ctk.CTkLabel(info_f, text=f"{lbl}: {val}", font=("Arial", 10), text_color="#3b8ed0" if lbl == "Λήξη Συμβολαίου" else "white")
                l.grid(row=r, column=c, padx=10, pady=1, sticky="w")
                c += 1
                if c > 2: c = 0; r += 1

    def export_to_excel(self):
        if self.filtered_df.empty: return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if path:
            self.filtered_df.to_excel(path, index=False)
            messagebox.showinfo("Success", "Η εξαγωγή ολοκληρώθηκε!")

    def render_page(self):
        for widget in self.results_area.winfo_children(): 
            widget.destroy()
        
        total_filtered = len(self.filtered_df)
        
        total_pages = max(1, (total_filtered // self.page_size) + (1 if total_filtered % self.page_size > 0 else 0))
        
        if self.current_page >= total_pages:
            self.current_page = total_pages - 1
        if self.current_page < 0:
            self.current_page = 0

        start_idx = self.current_page * self.page_size
        end_idx = start_idx + self.page_size
        page_data = self.filtered_df.iloc[start_idx:end_idx]
        
        for _, row in page_data.iterrows(): 
            self.create_card(row)
            
        self.page_label.configure(text=f"Σελίδα {self.current_page + 1} από {total_pages}")
        
        if total_filtered == 0:
            self.status_label.configure(text="❌ Δεν βρέθηκαν αποτελέσματα", text_color="firebrick")
        else:
            self.status_label.configure(text=f"✅ Εμφανίζονται {total_filtered} εγγραφές (Φιλτραρισμένες)", text_color="#27ae60")

    def open_multiselect(self, col, target, title, is_package=False):
        if self.df is None: return
        options = sorted([str(x) for x in self.df[col].dropna().unique() if str(x).strip() != ""])
        MultiSelectWindow(self, title, options, target, self.apply_filters, is_package)

    def next_page(self): self.current_page += 1; self.render_page()
    def prev_page(self): self.current_page = max(0, self.current_page - 1); self.render_page()
    def add_section_label(self, text, master=None): ctk.CTkLabel(master or self.sidebar, text=text, font=("Arial", 12, "bold"), text_color="#3b8ed0").pack(pady=(15, 2), padx=10, anchor="w")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    app = MegaEnergyCRM()
    app.mainloop()
