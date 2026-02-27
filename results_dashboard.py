"""
Federated Learning Live Results Dashboard
==========================================
Modern split-panel tkinter window with embedded matplotlib charts.

Layout:
  header bar + progress strip
  sidebar (experiment status list) | content notebook
    [Live Charts]  – accuracy + loss line charts, live-updated per round
                     click any legend swatch to show/hide that series
    [Round Table]  – per-round metric rows for selected experiment
    [Summary]      – final-results table + comparison bar charts

Thread-safety: worker -> GUI via a Queue, drained every 80 ms.
"""

import queue
from datetime import datetime
from typing import Dict, List, Optional


# Design tokens  
BG          = '#0d1021'   # page background
SURFACE     = '#141729'   # card surface
CARD        = '#1a1e35'   # slightly elevated card
BORDER      = '#252a47'   # subtle border
HDR_BG      = '#10142a'   # header bar background
HDR_FG      = '#e8ecf8'   # header primary text  
ACCENT      = '#6b8ef5'   # primary blue accent
GREEN       = '#4ec98b'   # success / honest
AMBER       = '#f0a832'   # in-progress / warning
RED         = '#e05572'   # error / compromised
DIM         = '#8b96b8'   # secondary text  (>=5:1 on SURFACE)
FG          = '#e0e4f2'   # main body text  (>=8:1 on SURFACE)
MUTED       = '#434c6e'   # separators, placeholders


SURFACE_CARD = CARD
BLUE         = ACCENT
HEADER       = HDR_BG

LINE_COLOURS = [
    '#6b8ef5', '#4ec98b', '#f0a832', '#e05572',
    '#b07ff5', '#5bc8e8', '#f07850', '#a8b0d0',
]

FONT_UI   = 'Segoe UI'
FONT_MONO = 'Consolas'

# Column tooltip descriptions
ROUND_COL_TIPS = {
    'Round':                 'Simulation round number',
    'Average Accuracy':      'Mean accuracy across all nodes',
    'Std Deviation':         'Standard deviation of accuracy across nodes',
    'Average Loss':          'Mean loss across all nodes',
    'Honest Accuracy':       'Mean accuracy of honest (non-attack) nodes',
    'Compromised Accuracy':  'Mean accuracy of compromised (Byzantine) nodes',
    'Drift Mean':            'Mean parameter drift across the network',
    'Drift Std':             'Standard deviation of parameter drift',
    'Consensus':             'Global consensus score \u2014 higher means more agreement',
    'Slope':                 'Regression slope of accuracy over recent window',
    'R\u00b2':                    'Coefficient of determination of accuracy regression',
    'Flags':                 'Number of nodes flagged as anomalous this round',
}

SUMMARY_COL_TIPS = {
    'Experiment':            'Experiment name / configuration',
    'Status':                'Completion status (Done / Failed)',
    'Final Accuracy':        'Final global model accuracy',
    'Honest Accuracy':       'Final accuracy of honest nodes',
    'Compromised Accuracy':  'Final accuracy of compromised nodes',
    'Attack Impact':         'Accuracy gap: honest \u2212 compromised',
    'TP':                    'True Positives \u2014 compromised nodes correctly flagged',
    'FP':                    'False Positives \u2014 honest nodes incorrectly flagged',
    'TN':                    'True Negatives \u2014 honest nodes correctly not flagged',
    'FN':                    'False Negatives \u2014 compromised nodes missed',
    'T_detect':              'Detection time: first round a true positive was flagged',
    'Duration':              'Total wall-clock time for the experiment',
}


class ResultsDashboard:

    def __init__(self, suite_name: str, experiment_names: List[str]):
        self.suite_name       = suite_name
        self.experiment_names = list(experiment_names)
        self._q: queue.Queue  = queue.Queue()
        self._root            = None
        self._round_data: Dict[str, List[dict]] = {n: [] for n in experiment_names}
        self._metrics_data: Dict[str, List[dict]] = {n: [] for n in experiment_names}
        self._status:     Dict[str, str]        = {n: 'waiting' for n in experiment_names}
        self._summary:    Dict[str, dict]       = {}
        self._current_exp: Optional[str]        = None
        self._done_count  = 0
        self._hidden_exps: set                  = set()   # click-to-toggle series

    #thread-safe notification API 
    def notify_start(self, exp_name: str, idx: int, total: int):
        self._q.put(('start', exp_name, idx, total))

    def notify_round(self, exp_name: str, row: dict):
        self._q.put(('round', exp_name, row))

    def notify_metrics(self, exp_name: str, rnd: int, metrics: dict):
        self._q.put(('metrics', exp_name, rnd, metrics))

    def notify_done(self, exp_name: str, result: dict):
        self._q.put(('done', exp_name, result))

    def notify_suite_done(self):
        self._q.put(('suite_done',))

    # -main entry
    def run(self):
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as e:
            print(f"Dashboard unavailable (tkinter missing): {e}")
            return

        # Set matplotlib backend before any pyplot import.
        # force=True ensures it takes effect even if matplotlib was already imported.
        try:
            import matplotlib
            matplotlib.use('TkAgg', force=True)
        except Exception:
            pass  

        self._tk   = tk
        self._ttk  = ttk
        # Enable DPI awareness for sharp text on high-DPI displays
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self._root = tk.Tk()
        try:
            self._build_gui()
        except Exception as e:
            import traceback
            print(f"Dashboard build error: {e}")
            traceback.print_exc()
            self._root.destroy()
            return
        self._root.after(80, self._drain_queue)
        self._root.mainloop()

    # GUI
    def _build_gui(self):
        tk, ttk, root = self._tk, self._ttk, self._root
        title = self.suite_name.replace('_', ' ').title()
        root.title(f"FL Dashboard  \u2013  {title}")
        root.geometry('1340x860')
        root.minsize(960, 600)
        root.configure(bg=BG)

        # Style sheet
        st = ttk.Style(root)
        st.theme_use('clam')

        st.configure('D.TNotebook',
                     background=SURFACE, borderwidth=0, tabmargins=[0, 0, 0, 0])
        st.configure('D.TNotebook.Tab',
                     background=CARD, foreground=DIM,
                     padding=[18, 8], font=(FONT_UI, 10), borderwidth=0)
        st.map('D.TNotebook.Tab',
               background=[('selected', SURFACE), ('active', '#22274a')],
               foreground=[('selected', HDR_FG),   ('active', FG)])

        st.configure('D.Treeview',
                     background=CARD, foreground=FG,
                     fieldbackground=CARD, rowheight=28,
                     font=(FONT_MONO, 10), borderwidth=0)
        st.configure('D.Treeview.Heading',
                     background=BORDER, foreground=ACCENT,
                     font=(FONT_UI, 10, 'bold'), relief='flat')
        st.map('D.Treeview', background=[('selected', '#252b50')])

        for sname in ('D.Vertical.TScrollbar', 'D.Horizontal.TScrollbar'):
            st.configure(sname, background=BORDER, troughcolor=SURFACE,
                         arrowcolor=DIM, borderwidth=0, width=8)

        # The Header
        hdr = tk.Frame(root, bg=HDR_BG, height=62)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)

        left_hdr = tk.Frame(hdr, bg=HDR_BG)
        left_hdr.pack(side='left', fill='y', padx=(20, 0))
        tk.Label(left_hdr,
                 text='Federated Learning Dashboard',
                 font=(FONT_UI, 13, 'bold'), bg=HDR_BG, fg=HDR_FG
                 ).pack(side='top', anchor='w', pady=(12, 0))
        tk.Label(left_hdr,
                 text=self.suite_name.replace('_', ' '),
                 font=(FONT_UI, 9), bg=HDR_BG, fg=DIM
                 ).pack(side='top', anchor='w')

        right_hdr = tk.Frame(hdr, bg=HDR_BG)
        right_hdr.pack(side='right', fill='y', padx=(0, 20))
        tk.Label(right_hdr,
                 text=f"Started  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
                 font=(FONT_UI, 9), bg=HDR_BG, fg=DIM
                 ).pack(side='top', anchor='e', pady=(14, 0))
        self._ts_lbl = tk.Label(right_hdr, text='',
                                font=(FONT_UI, 9), bg=HDR_BG, fg=DIM)
        self._ts_lbl.pack(side='top', anchor='e')

        # Progress bar and accent strip
        tk.Frame(root, bg=ACCENT, height=2).pack(fill='x')
        self._prog_c = tk.Canvas(root, bg=SURFACE, height=22,
                                 highlightthickness=0)
        self._prog_c.pack(fill='x')
        self._prog_bg = self._prog_c.create_rectangle(
            0, 0, 4000, 22, fill=BORDER, outline='')
        self._prog_r = self._prog_c.create_rectangle(
            0, 0, 0, 22, fill=ACCENT, outline='')
        self._prog_txt = self._prog_c.create_text(
            670, 11, text='0 %', fill=HDR_FG,
            font=(FONT_UI, 9, 'bold'))

        # Status row
        sr = tk.Frame(root, bg=BG)
        sr.pack(fill='x', padx=18, pady=(8, 0))
        tk.Label(sr, text='STATUS',
                 font=(FONT_UI, 8, 'bold'), bg=BG, fg=MUTED
                 ).pack(side='left')
        self._active_lbl = tk.Label(sr, text='  Initialising\u2026',
                                    font=(FONT_UI, 10), bg=BG, fg=AMBER)
        self._active_lbl.pack(side='left', padx=8)

        # Body
        body = tk.Frame(root, bg=BG)
        body.pack(fill='both', expand=True, pady=(8, 0))

        # sidebar
        sb = tk.Frame(body, bg=SURFACE, width=244)
        sb.pack(side='left', fill='y', padx=(12, 0), pady=(0, 12))
        sb.pack_propagate(False)

        sb_head = tk.Frame(sb, bg=SURFACE)
        sb_head.pack(fill='x', padx=14, pady=(14, 6))
        tk.Label(sb_head, text='EXPERIMENTS',
                 font=(FONT_UI, 8, 'bold'), bg=SURFACE, fg=MUTED
                 ).pack(side='left')
        self._exp_count_lbl = tk.Label(
            sb_head, text=f'0 / {len(self.experiment_names)}',
            font=(FONT_UI, 8), bg=SURFACE, fg=DIM)
        self._exp_count_lbl.pack(side='right')

        tk.Frame(sb, bg=BORDER, height=1).pack(fill='x', padx=10)

        sb_inner = tk.Frame(sb, bg=SURFACE)
        sb_inner.pack(fill='both', expand=True, padx=6, pady=6)
        self._sb_items: Dict[str, dict] = {}

        for name in self.experiment_names:
            row = tk.Frame(sb_inner, bg=SURFACE, pady=1)
            row.pack(fill='x')
            dot = tk.Label(row, text='\u25cf',
                           font=(FONT_UI, 9), bg=SURFACE, fg=MUTED)
            dot.pack(side='left', padx=(8, 4))
            lbl = tk.Label(row, text=name[:28],
                           font=(FONT_UI, 9), bg=SURFACE, fg=DIM, anchor='w')
            lbl.pack(side='left', fill='x', expand=True)
            ico = tk.Label(row, text='',
                           font=(FONT_UI, 8, 'bold'), bg=SURFACE, fg=DIM,
                           width=4)
            ico.pack(side='right', padx=6)
            self._sb_items[name] = {'row': row, 'dot': dot, 'lbl': lbl, 'ico': ico}

        # content
        content = tk.Frame(body, bg=BG)
        content.pack(side='left', fill='both', expand=True,
                     padx=10, pady=(0, 12))
        nb = ttk.Notebook(content, style='D.TNotebook')
        nb.pack(fill='both', expand=True)
        self._nb = nb

        t1 = tk.Frame(nb, bg=SURFACE); nb.add(t1, text='   Live Charts   ')
        t2 = tk.Frame(nb, bg=SURFACE); nb.add(t2, text='   Round Table   ')
        t3 = tk.Frame(nb, bg=SURFACE); nb.add(t3, text='   Summary   ')
        t4 = tk.Frame(nb, bg=SURFACE); nb.add(t4, text='   Advanced Metrics   ')
        self._build_charts_tab(t1)
        self._build_table_tab(t2)
        self._build_summary_tab(t3)
        self._build_advanced_tab(t4)

    # tabs 
    def _build_charts_tab(self, parent):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        fig.patch.set_facecolor(SURFACE)
        self._ax_acc, self._ax_loss = axes
        for ax, title, ylabel in [
            (self._ax_acc,  'Accuracy over Rounds', 'Accuracy'),
            (self._ax_loss, 'Loss over Rounds',     'Loss'),
        ]:
            ax.set_facecolor(BG)
            ax.set_title(title, color=HDR_FG, fontsize=11,
                         fontfamily=FONT_UI, pad=10, fontweight='semibold')
            ax.set_xlabel('Round',  color=DIM, fontsize=9, fontfamily=FONT_UI)
            ax.set_ylabel(ylabel,   color=DIM, fontsize=9, fontfamily=FONT_UI)
            ax.tick_params(colors=DIM)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, linestyle='--', linewidth=0.55, alpha=0.9)
        fig.tight_layout(pad=2.6)
        self._fig    = fig
        self._canvas = FigureCanvasTkAgg(fig, master=parent)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=(10, 4))

        # hover state
        self._ann_acc   = None
        self._ann_loss  = None
        self._loss_ymax = 0.0

        # Clickable series legend (allows you to hide/show individual graphs on the charts)
        self._hidden_exps:   set          = set()
        self._hover_data:    Dict         = {}
        self._legend_btns:   Dict[str, dict] = {}

        legend_outer = self._tk.Frame(parent, bg=SURFACE)
        legend_outer.pack(fill='x', padx=10, pady=(0, 8))
        self._tk.Label(legend_outer, text='SERIES',
                       font=(FONT_UI, 8, 'bold'), bg=SURFACE, fg=MUTED
                       ).pack(side='left', padx=(6, 10))
        legend_scroll = self._tk.Frame(legend_outer, bg=SURFACE)
        legend_scroll.pack(side='left', fill='x', expand=True)

        for i, name in enumerate(self.experiment_names):
            colour = LINE_COLOURS[i % len(LINE_COLOURS)]
            self._add_legend_btn(legend_scroll, name, colour)

        self._setup_hover()

    def _add_legend_btn(self, parent, name: str, colour: str):
        """One clickable swatch + label for the series legend."""
        tk = self._tk
        is_visible = name not in self._hidden_exps

        container = tk.Frame(parent, bg=SURFACE, padx=4, pady=2, cursor='hand2')
        container.pack(side='left', padx=2)

        sw = tk.Canvas(container, width=12, height=12,
                       bg=SURFACE, highlightthickness=0)
        sw.pack(side='left', padx=(0, 4))
        rect_id = sw.create_rectangle(1, 1, 11, 11,
                                       fill=colour if is_visible else MUTED,
                                       outline='')

        lbl = tk.Label(container, text=name[:20],
                       font=(FONT_UI, 9), bg=SURFACE,
                       fg=FG if is_visible else MUTED)
        lbl.pack(side='left')

        def _toggle(event, n=name):
            self._toggle_series(n)

        for widget in (container, sw, lbl):
            widget.bind('<Button-1>', _toggle)

        self._legend_btns[name] = {
            'container': container, 'swatch': sw,
            'rect_id': rect_id, 'label': lbl, 'colour': colour,
        }

    def _toggle_series(self, name: str):
        """Show/hide a series and refresh all charts."""
        if name in self._hidden_exps:
            self._hidden_exps.discard(name)
        else:
            self._hidden_exps.add(name)
        self._refresh_legend_btn(name)
        # Also refresh the advanced-tab legend if it exists
        adv_item = getattr(self, '_adv_legend_btns', {}).get(name)
        if adv_item:
            is_vis = name not in self._hidden_exps
            adv_item['swatch'].itemconfig(
                adv_item['rect_id'],
                fill=adv_item['colour'] if is_vis else MUTED)
            adv_item['label'].config(fg=FG if is_vis else MUTED)
        self._update_live_charts()
        self._update_advanced_charts()

    def _refresh_legend_btn(self, name: str):
        item = self._legend_btns.get(name)
        if not item:
            return
        is_visible = name not in self._hidden_exps
        item['swatch'].itemconfig(item['rect_id'],
                                   fill=item['colour'] if is_visible else MUTED)
        item['label'].config(fg=FG if is_visible else MUTED)

    def _build_table_tab(self, parent):
        tk, ttk = self._tk, self._ttk
        sr = tk.Frame(parent, bg=SURFACE)
        sr.pack(fill='x', padx=12, pady=(12, 6))
        tk.Label(sr, text='Experiment:',
                 font=(FONT_UI, 10), bg=SURFACE, fg=FG
                 ).pack(side='left', padx=(0, 8))
        self._table_var = tk.StringVar(value=self.experiment_names[0])
        ttk.OptionMenu(sr, self._table_var,
                       self.experiment_names[0], *self.experiment_names,
                       command=self._on_table_select).pack(side='left')
        tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', padx=10, pady=(0, 4))
        cols = (
            'Round',
            'Average Accuracy',
            'Std Deviation',
            'Average Loss',
            'Honest Accuracy',
            'Compromised Accuracy',
            'Drift Mean',
            'Drift Std',
            'Consensus',
            'Slope',
            'R²',
            'Flags',
        )
        self._round_tree = self._make_tree(
            parent, cols,
            col_width=120,
            col_widths={'Round': 60, 'Flags': 55, 'R²': 70, 'Slope': 80,
                        'Consensus': 95, 'Drift Mean': 95, 'Drift Std': 85},
        )
        self._bind_tree_tooltips(self._round_tree, ROUND_COL_TIPS)
    def _build_summary_tab(self, parent):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        tk = self._tk
        hdr_row = tk.Frame(parent, bg=SURFACE)
        hdr_row.pack(fill='x', padx=14, pady=(12, 6))
        tk.Label(hdr_row, text='Experiment Results',
                 font=(FONT_UI, 11, 'bold'), bg=SURFACE, fg=HDR_FG
                 ).pack(side='left')
        tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', padx=10, pady=(0, 6))
        cols = (
            'Experiment',
            'Status',
            'Final Accuracy',
            'Honest Accuracy',
            'Compromised Accuracy',
            'Attack Impact',
            'TP',
            'FP',
            'TN',
            'FN',
            'T_detect',
            'Duration',
        )
        self._sum_tree = self._make_tree(
            parent, cols,
            col_widths={'Experiment': 200, 'Status': 62,
                        'Final Accuracy': 110, 'Honest Accuracy': 112,
                        'Compromised Accuracy': 146,
                        'Attack Impact': 100,
                        'TP': 44, 'FP': 44, 'TN': 44, 'FN': 44,
                        'T_detect': 68, 'Duration': 78},
            height=7)
        self._sum_iids: dict = {}
        for name in self.experiment_names:
            iid = self._sum_tree.insert(
                '', 'end',
                values=(name, 'Waiting', '–', '–', '–', '–', '–'))
            self._sum_iids[name] = iid
        self._bind_tree_tooltips(self._sum_tree, SUMMARY_COL_TIPS)
        cf = tk.Frame(parent, bg=SURFACE)
        cf.pack(fill='both', expand=True, padx=8, pady=6)
        self._sum_fig, self._sum_axes = plt.subplots(1, 2, figsize=(11, 3.0))
        self._sum_fig.patch.set_facecolor(SURFACE)
        for ax in self._sum_axes:
            ax.set_facecolor(BG)
            ax.tick_params(colors=DIM)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, axis='y',
                    linestyle='--', linewidth=0.55, alpha=0.9)
        self._sum_fig.tight_layout(pad=2)
        self._sum_canvas = FigureCanvasTkAgg(self._sum_fig, master=cf)
        self._sum_canvas.draw()
        self._sum_canvas.get_tk_widget().pack(fill='both', expand=True)

    def _make_tree(self, parent, columns, col_width=140,
                   col_widths=None, height=None):
        tk, ttk = self._tk, self._ttk
        wrap = tk.Frame(parent, bg=SURFACE)
        wrap.pack(fill='both', expand=True, padx=10, pady=(0, 8))
        kw = dict(columns=columns, show='headings', style='D.Treeview')
        if height:
            kw['height'] = height
        tree = ttk.Treeview(wrap, **kw)
        for col in columns:
            w = (col_widths or {}).get(col, col_width)
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor='center', minwidth=50)
        vs = ttk.Scrollbar(wrap, orient='vertical', command=tree.yview,
                           style='D.Vertical.TScrollbar')
        hs = ttk.Scrollbar(wrap, orient='horizontal', command=tree.xview,
                           style='D.Horizontal.TScrollbar')
        tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vs.grid(row=0, column=1, sticky='ns')
        hs.grid(row=1, column=0, sticky='ew')
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        return tree

    def _bind_tree_tooltips(self, tree, descriptions: dict):
        """Bind hover tooltips to Treeview column headings."""
        tk = self._tk
        tip_win = [None]
        after_id = [None]
        cur_col = [None]

        def _show(x, y, col, desc):
            _hide_tip()
            tw = tk.Toplevel(tree)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f'+{x + 12}+{y + 16}')
            tw.configure(bg=BORDER)
            inner = tk.Frame(tw, bg=CARD, padx=10, pady=6)
            inner.pack(padx=1, pady=1)
            tk.Label(inner, text=col, font=(FONT_UI, 9, 'bold'),
                     bg=CARD, fg=ACCENT).pack(anchor='w')
            tk.Label(inner, text=desc, font=(FONT_UI, 8),
                     bg=CARD, fg=FG, wraplength=260,
                     justify='left').pack(anchor='w', pady=(2, 0))
            tip_win[0] = tw

        def _hide_tip():
            if tip_win[0]:
                tip_win[0].destroy()
                tip_win[0] = None

        def _on_motion(event):
            region = tree.identify_region(event.x, event.y)
            if region == 'heading':
                col_id = tree.identify_column(event.x)
                if col_id == '#0':
                    _cancel()
                    return
                col_idx = int(col_id.replace('#', '')) - 1
                cols = tree['columns']
                if 0 <= col_idx < len(cols):
                    col_name = cols[col_idx]
                    if col_name != cur_col[0]:
                        _cancel()
                        cur_col[0] = col_name
                        desc = descriptions.get(col_name)
                        if desc:
                            after_id[0] = tree.after(
                                420,
                                lambda ex=event.x_root, ey=event.y_root:
                                    _show(ex, ey, col_name, desc))
            else:
                _cancel()

        def _cancel(event=None):
            cur_col[0] = None
            if after_id[0]:
                tree.after_cancel(after_id[0])
                after_id[0] = None
            _hide_tip()

        tree.bind('<Motion>', _on_motion)
        tree.bind('<Leave>', _cancel)

    # Draining the Queue
    def _drain_queue(self):
        try:
            while True:
                msg   = self._q.get_nowait()
                mtype = msg[0]
                if mtype == 'start':
                    _, name, idx, total = msg
                    self._current_exp  = name
                    self._status[name] = 'running'
                    self._active_lbl.config(
                        text=f'  [{idx}/{total}]  Running: {name}', fg=AMBER)
                    self._update_sidebar(name, 'running')
                    self._table_var.set(name)
                    self._refresh_round_table(name)
                    self._set_progress((idx - 1) / total)
                elif mtype == 'round':
                    _, name, row = msg
                    self._round_data[name].append(row)
                    if name == self._table_var.get():
                        self._insert_round_row(row)
                    self._update_live_charts()
                elif mtype == 'metrics':
                    _, name, rnd, metrics = msg
                    self._metrics_data[name].append(
                        dict(metrics, round=rnd))
                    # Merge metrics into the last matching round row for the table
                    for r in reversed(self._round_data.get(name, [])):
                        if r.get('round') == rnd:
                            r.update(metrics)
                            break
                    # Refresh round table if this experiment is selected
                    if name == self._table_var.get():
                        self._refresh_round_table(name)
                    self._update_advanced_charts()
                elif mtype == 'done':
                    _, name, res = msg
                    self._status[name]  = 'failed' if res.get('error') else 'done'
                    self._summary[name] = res
                    self._done_count   += 1
                    # Pull detection metrics from the last metrics row
                    mdata = self._metrics_data.get(name, [])
                    if mdata and 'detection' not in res:
                        last = mdata[-1]
                        res['detection'] = {
                            'true_positives':  last.get('tp', 0),
                            'false_positives': last.get('fp', 0),
                            'true_negatives':  last.get('tn', 0),
                            'false_negatives': last.get('fn', 0),
                            'detection_time':  None,
                        }
                        # Find first round where tp > 0
                        for md in mdata:
                            if md.get('tp', 0) > 0:
                                res['detection']['detection_time'] = md['round']
                                break
                    self._update_sidebar(name, self._status[name])
                    self._update_sum_row(name, res)
                    self._set_progress(
                        self._done_count / len(self.experiment_names))
                    self._refresh_summary_chart()
                elif mtype == 'suite_done':
                    self._active_lbl.config(
                        text='  All experiments complete', fg=GREEN)
                    self._ts_lbl.config(
                        text=f"Finished  "
                             f"{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
                    self._set_progress(1.0)
        except queue.Empty:
            pass
        if self._root:
            self._root.after(80, self._drain_queue)

    # Progress bar
    def _set_progress(self, frac: float):
        w = self._prog_c.winfo_width() or 1300
        fill_w = int(w * max(0.005, frac))
        self._prog_c.coords(self._prog_r, 0, 0, fill_w, 22)
        pct = int(frac * 100)
        self._prog_c.itemconfig(self._prog_txt, text=f'{pct} %')
        self._prog_c.coords(self._prog_txt, w // 2, 11)

    # Side bar helpers
    def _update_sidebar(self, name: str, status: str):
        item = self._sb_items.get(name)
        if not item:
            return
        ACTIVE_BG = '#1a1e35'
        cfg = { 
            'waiting': (SURFACE,    MUTED,  '\u25cf', ''),
            'running': (ACTIVE_BG,  AMBER,  '\u25cf', 'RUN'),
            'done':    (ACTIVE_BG,  GREEN,  '\u2714', ' OK'),
            'failed':  (ACTIVE_BG,  RED,    '\u2718', 'ERR'),
        }
        bg, col, dot_ch, ico_ch = cfg.get(status, (SURFACE, MUTED, '\u25cf', ''))
        lbl_fg = HDR_FG if status != 'waiting' else DIM
        item['row'].config(bg=bg)
        item['dot'].config(bg=bg, fg=col, text=dot_ch)
        item['lbl'].config(bg=bg, fg=lbl_fg)
        item['ico'].config(bg=bg, fg=col, text=ico_ch)
        done = sum(1 for s in self._status.values() if s == 'done')
        self._exp_count_lbl.config(text=f'{done} / {len(self.experiment_names)}')

    def _setup_hover(self):
        """Attach a single motion-notify handler to the figure canvas."""
        self._hover_data: Dict = {}   # ax -> list of (xs, ys, label, colour)
        self._canvas.mpl_connect('motion_notify_event', self._on_hover)

    def _on_hover(self, event):
        #Annotations for each data point on the charts.
        import numpy as np

        ax_map = {
            self._ax_acc:  ('_ann_acc',  '{:.4f}', 'Accuracy'),
            self._ax_loss: ('_ann_loss', '{:.4f}', 'Loss'),
        }
        if event.inaxes not in ax_map:
            return

        ax       = event.inaxes
        ann_attr, fmt, metric = ax_map[ax]

        data_series = self._hover_data.get(id(ax), [])
        if not data_series:
            return

        # Find closest point across all series (display-coordinate distance)
        best_dist = float('inf')
        best_x = best_y = best_label = best_colour = None

        try:
            # Transform axes data -> display pixels for distance calculation
            xy_disp = ax.transData.transform
            cx, cy  = xy_disp((event.xdata, event.ydata))

            for xs, ys, label, colour in data_series:
                for x, y in zip(xs, ys):
                    px, py = xy_disp((x, y))
                    dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist  = dist
                        best_x, best_y = x, y
                        best_label = label
                        best_colour = colour
        except Exception:
            return

        SNAP_PX = 22  # pixels — only show tooltip if this close

        # Remove old annotation regardless
        old = getattr(self, ann_attr, None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
            setattr(self, ann_attr, None)

        if best_dist > SNAP_PX:
            self._canvas.draw_idle()
            return

        text = f"{best_label}\nRound {int(best_x)}\n{metric}: {fmt.format(best_y)}"

        ann = ax.annotate(
            text,
            xy=(best_x, best_y),
            xytext=(14, 14),
            textcoords='offset points',
            fontsize=9,
            fontfamily=FONT_UI,
            color=HDR_FG,
            bbox=dict(
                boxstyle='round,pad=0.55',
                facecolor=CARD,
                edgecolor=best_colour,
                linewidth=1.5,
                alpha=0.97,
            ),
            arrowprops=dict(
                arrowstyle='->',
                color=best_colour,
                lw=1.2,
            ),
            zorder=10,
        )
        setattr(self, ann_attr, ann)
        self._canvas.draw_idle()

    # Live charts 
    def _update_live_charts(self):
        ax_a, ax_l = self._ax_acc, self._ax_loss

        # Invalidate hover annotations before clearing axes
        self._ann_acc  = None
        self._ann_loss = None
        self._hover_data = {}

        ax_a.cla(); ax_l.cla()
        for ax, title, ylabel in [
            (ax_a, 'Accuracy over Rounds', 'Accuracy'),
            (ax_l, 'Loss over Rounds',     'Loss'),
        ]:
            ax.set_facecolor(BG)
            ax.set_title(title, color=HDR_FG, fontsize=11,
                         fontfamily=FONT_UI, pad=8, fontweight='semibold')
            ax.set_xlabel('Round',  color=DIM, fontsize=9, fontfamily=FONT_UI)
            ax.set_ylabel(ylabel,   color=DIM, fontsize=9, fontfamily=FONT_UI)
            ax.tick_params(colors=DIM, labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, linestyle='--', linewidth=0.55, alpha=0.9)

        # Accuracy axis: always fixed 0 -> 1 so scale never jumps
        ax_a.set_ylim(0.0, 1.05)

        drawn = False
        for i, name in enumerate(self.experiment_names):
            data = self._round_data[name]
            if not data:
                continue
            c      = LINE_COLOURS[i % len(LINE_COLOURS)]
            rs     = [d['round']        for d in data]
            accs   = [d['avg_accuracy'] for d in data]
            losses = [d['avg_loss']     for d in data]
            hidden = name in self._hidden_exps
            lw     = 2.0 if self._status[name] == 'running' else 1.6
            alpha  = 0.07 if hidden else (
                     1.0  if self._status[name] in ('running', 'done') else 0.5)
            short  = name[:22]
            kw = dict(color=c, lw=lw, alpha=alpha,
                      label=short, marker='o', markersize=3.8,
                      markerfacecolor=c, markeredgewidth=0)
            ax_a.plot(rs, accs,   **kw)
            ax_l.plot(rs, losses, **kw)

            # Only register visible series for hover
            if not hidden:
                self._hover_data.setdefault(id(ax_a), []).append((rs, accs,   short, c))
                self._hover_data.setdefault(id(ax_l), []).append((rs, losses, short, c))
                if losses:
                    self._loss_ymax = max(self._loss_ymax, max(losses) * 1.12)

            drawn = True

        if drawn:
            # Stable loss scale: only ever grows, never shrinks
            ax_l.set_ylim(bottom=0, top=max(self._loss_ymax, 0.5))
            for ax in (ax_a, ax_l):
                leg = ax.legend(fontsize=8.5,
                                facecolor=CARD, edgecolor=BORDER,
                                labelcolor=FG, loc='best', framealpha=0.92)
                for line in leg.get_lines():
                    line.set_linewidth(2.0)

        self._fig.tight_layout(pad=2.6)
        self._canvas.draw_idle()

    # Round table helpers
    def _refresh_round_table(self, name: str):
        self._round_tree.delete(*self._round_tree.get_children())
        for row in self._round_data.get(name, []):
            self._insert_round_row(row)

    def _insert_round_row(self, row: dict):
        iid = self._round_tree.insert('', 'end', values=(
            row.get('round', ''),
            self._fmt(row.get('avg_accuracy')),
            self._fmt(row.get('std_accuracy')),
            self._fmt(row.get('avg_loss')),
            self._fmt(row.get('honest_accuracy')),
            self._fmt(row.get('compromised_accuracy')),
            self._fmt(row.get('drift_mean')),
            self._fmt(row.get('drift_std')),
            self._fmt(row.get('consensus')),
            self._fmt(row.get('slope')),
            self._fmt(row.get('r_squared')),
            row.get('n_flagged', '\u2013'),
        ))
        self._round_tree.see(iid)

    def _on_table_select(self, name):
        self._refresh_round_table(name)

    # SUmmary tab helpers
    def _update_sum_row(self, name: str, res: dict):
        iid = self._sum_iids.get(name)
        if not iid:
            return
        det = res.get('detection', {}) or {}
        self._sum_tree.item(iid, values=(
            name,
            'Failed' if res.get('error') else 'Done',
            self._fmt(res.get('final_accuracy')),
            self._fmt(res.get('honest_accuracy')),
            self._fmt(res.get('compromised_accuracy')),
            self._fmt(res.get('attack_impact')),
            det.get('true_positives', '\u2013'),
            det.get('false_positives', '\u2013'),
            det.get('true_negatives', '\u2013'),
            det.get('false_negatives', '\u2013'),
            det.get('detection_time', '\u2013') if det.get('detection_time') is not None else '\u2013',
            res.get('duration', '-'),
        ))

    def _refresh_summary_chart(self):
        import numpy as np
        done = [(n, self._summary[n]) for n in self.experiment_names
                if n in self._summary and not self._summary[n].get('error')]
        if not done:
            return
        names  = [d[0][:18] for d in done]
        accs   = [d[1].get('final_accuracy')      or 0 for d in done]
        honest = [d[1].get('honest_accuracy')      or 0 for d in done]
        compr  = [d[1].get('compromised_accuracy') or 0 for d in done]
        x      = np.arange(len(names))
        w      = 0.30
        cols   = LINE_COLOURS[:len(names)]
        ax_a, ax_b = self._sum_axes
        ax_a.cla(); ax_b.cla()
        for ax, title in [
            (ax_a, 'Final Accuracy by Experiment'),
            (ax_b, 'Honest vs Compromised Accuracy'),
        ]:
            ax.set_facecolor(BG)
            ax.set_title(title, color=HDR_FG, fontsize=10,
                         fontfamily=FONT_UI, pad=8, fontweight='semibold')
            ax.tick_params(colors=DIM, labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, axis='y',
                    linestyle='--', linewidth=0.5, alpha=0.9)
        bars = ax_a.bar(x, accs, color=cols, edgecolor=BG, width=0.5)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(names, rotation=18, ha='right', color=DIM, fontsize=8)
        for bar, v in zip(bars, accs):
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                      f'{v:.3f}', ha='center', va='bottom',
                      color=FG, fontsize=8, fontfamily=FONT_UI)
        ax_b.bar(x - w/2, honest, w, label='Honest Accuracy',
                 color=GREEN, edgecolor=BG, alpha=0.9)
        ax_b.bar(x + w/2, compr,  w, label='Compromised Accuracy',
                 color=RED,   edgecolor=BG, alpha=0.9)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(names, rotation=18, ha='right', color=DIM, fontsize=8)
        ax_b.legend(fontsize=8.5, facecolor=CARD, edgecolor=BORDER,
                    labelcolor=FG, framealpha=0.92)
        self._sum_fig.tight_layout(pad=2)
        self._sum_canvas.draw_idle()

    # Advanced metrics tab (5 charts: drift, consensus, slope, R², flags)
    def _build_advanced_tab(self, parent):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(12, 7.5))
        fig.patch.set_facecolor(SURFACE)
        gs = GridSpec(2, 6, figure=fig, hspace=0.42, wspace=0.65)

        self._adv_ax_drift     = fig.add_subplot(gs[0, 0:2])
        self._adv_ax_consensus = fig.add_subplot(gs[0, 2:4])
        self._adv_ax_slope     = fig.add_subplot(gs[0, 4:6])
        self._adv_ax_r2        = fig.add_subplot(gs[1, 0:3])
        self._adv_ax_flags     = fig.add_subplot(gs[1, 3:6])

        titles = [
            (self._adv_ax_drift,     'Drift per Round',            'Drift'),
            (self._adv_ax_consensus, 'Consensus & Peer Dev',       'Value'),
            (self._adv_ax_slope,     'Regression Slope',           'Slope'),
            (self._adv_ax_r2,        'R\u00b2 (Goodness of Fit)',        'R\u00b2'),
            (self._adv_ax_flags,     'Detection Flags per Round',  'Count'),
        ]
        for ax, title, ylabel in titles:
            ax.set_facecolor(BG)
            ax.set_title(title, color=HDR_FG, fontsize=10,
                         fontfamily=FONT_UI, pad=8, fontweight='semibold')
            ax.set_xlabel('Round', color=DIM, fontsize=8, fontfamily=FONT_UI)
            ax.set_ylabel(ylabel,  color=DIM, fontsize=8, fontfamily=FONT_UI)
            ax.tick_params(colors=DIM, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, linestyle='--', linewidth=0.5, alpha=0.9)

        self._adv_fig = fig
        self._adv_canvas = FigureCanvasTkAgg(fig, master=parent)
        self._adv_canvas.draw()
        self._adv_canvas.get_tk_widget().pack(fill='both', expand=True,
                                               padx=10, pady=(8, 2))

        # Descriptions under each graph
        desc_frame = self._tk.Frame(parent, bg=SURFACE)
        desc_frame.pack(fill='x', padx=14, pady=(0, 2))
        descriptions = [
            ('Drift',
             'Mean parameter drift \u00b1 std dev across all nodes.'),
            ('Consensus',
             'Global agreement score and mean peer deviation.'),
            ('Slope',
             'Regression slope of accuracy over a sliding window.'),
            ('R\u00b2',
             'Coefficient of determination \u2014 near 1.0 = strong trend.'),
            ('Flags',
             'Number of nodes flagged as anomalous each round.'),
        ]
        for title, desc in descriptions:
            d = self._tk.Frame(desc_frame, bg=SURFACE)
            d.pack(side='left', fill='x', expand=True, padx=4)
            self._tk.Label(d, text=f'{title}:',
                           font=(FONT_UI, 8, 'bold'),
                           bg=SURFACE, fg=ACCENT).pack(side='left')
            self._tk.Label(d, text=f' {desc}',
                           font=(FONT_UI, 7),
                           bg=SURFACE, fg=DIM, wraplength=200,
                           justify='left').pack(side='left', fill='x')

        # Series legend (toggle show/hide, shared with live charts)
        self._adv_legend_btns: Dict[str, dict] = {}
        legend_outer = self._tk.Frame(parent, bg=SURFACE)
        legend_outer.pack(fill='x', padx=10, pady=(2, 6))
        self._tk.Label(legend_outer, text='SERIES',
                       font=(FONT_UI, 8, 'bold'), bg=SURFACE, fg=MUTED
                       ).pack(side='left', padx=(6, 10))
        legend_scroll = self._tk.Frame(legend_outer, bg=SURFACE)
        legend_scroll.pack(side='left', fill='x', expand=True)
        for i, name in enumerate(self.experiment_names):
            colour = LINE_COLOURS[i % len(LINE_COLOURS)]
            self._add_adv_legend_btn(legend_scroll, name, colour)

        # Hover annotations for advanced charts
        self._ann_drift     = None
        self._ann_consensus = None
        self._ann_slope     = None
        self._ann_r2        = None
        self._ann_flags     = None
        self._adv_hover_data: Dict = {}
        self._adv_canvas.mpl_connect('motion_notify_event', self._on_adv_hover)

    def _add_adv_legend_btn(self, parent, name: str, colour: str):
        """One clickable swatch + label for the advanced-tab series legend."""
        tk = self._tk
        is_visible = name not in self._hidden_exps
        container = tk.Frame(parent, bg=SURFACE, padx=4, pady=2, cursor='hand2')
        container.pack(side='left', padx=2)
        sw = tk.Canvas(container, width=12, height=12,
                       bg=SURFACE, highlightthickness=0)
        sw.pack(side='left', padx=(0, 4))
        rect_id = sw.create_rectangle(1, 1, 11, 11,
                                       fill=colour if is_visible else MUTED,
                                       outline='')
        lbl = tk.Label(container, text=name[:20],
                       font=(FONT_UI, 9), bg=SURFACE,
                       fg=FG if is_visible else MUTED)
        lbl.pack(side='left')

        def _toggle(event, n=name):
            self._toggle_series(n)
        for widget in (container, sw, lbl):
            widget.bind('<Button-1>', _toggle)

        self._adv_legend_btns[name] = {
            'container': container, 'swatch': sw,
            'rect_id': rect_id, 'label': lbl, 'colour': colour,
        }

    def _on_adv_hover(self, event):
        """Hover tooltip for advanced metrics charts."""
        ax_map = {
            self._adv_ax_drift:     ('_ann_drift',     '{:.4f}', 'Drift'),
            self._adv_ax_consensus: ('_ann_consensus', '{:.4f}', 'Value'),
            self._adv_ax_slope:     ('_ann_slope',     '{:.6f}', 'Slope'),
            self._adv_ax_r2:        ('_ann_r2',        '{:.4f}', 'R\u00b2'),
            self._adv_ax_flags:     ('_ann_flags',     '{:.0f}', 'Flagged'),
        }
        if event.inaxes not in ax_map:
            return

        ax = event.inaxes
        ann_attr, fmt, metric = ax_map[ax]
        data_series = self._adv_hover_data.get(id(ax), [])
        if not data_series:
            return

        best_dist = float('inf')
        best_x = best_y = best_label = best_colour = None

        try:
            xy_disp = ax.transData.transform
            cx, cy = xy_disp((event.xdata, event.ydata))
            for xs, ys, label, colour in data_series:
                for x, y in zip(xs, ys):
                    px, py = xy_disp((x, y))
                    dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_x, best_y = x, y
                        best_label = label
                        best_colour = colour
        except Exception:
            return

        SNAP_PX = 22
        old = getattr(self, ann_attr, None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
            setattr(self, ann_attr, None)

        if best_dist > SNAP_PX:
            self._adv_canvas.draw_idle()
            return

        text = f"{best_label}\nRound {int(best_x)}\n{metric}: {fmt.format(best_y)}"
        ann = ax.annotate(
            text,
            xy=(best_x, best_y),
            xytext=(14, 14),
            textcoords='offset points',
            fontsize=8,
            fontfamily=FONT_UI,
            color=HDR_FG,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=CARD,
                      edgecolor=best_colour, linewidth=1.5, alpha=0.97),
            arrowprops=dict(arrowstyle='->', color=best_colour, lw=1.2),
            zorder=10,
        )
        setattr(self, ann_attr, ann)
        self._adv_canvas.draw_idle()

    def _update_advanced_charts(self):
        import numpy as np

        axes = [self._adv_ax_drift, self._adv_ax_consensus,
                self._adv_ax_slope, self._adv_ax_r2, self._adv_ax_flags]

        # Invalidate hover annotations before clearing
        for attr in ('_ann_drift', '_ann_consensus', '_ann_slope',
                     '_ann_r2', '_ann_flags'):
            setattr(self, attr, None)
        self._adv_hover_data = {}

        for ax in axes:
            ax.cla()
            ax.set_facecolor(BG)
            ax.tick_params(colors=DIM, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, linestyle='--', linewidth=0.5, alpha=0.9)

        titles = [
            (self._adv_ax_drift,     'Drift per Round',            'Drift'),
            (self._adv_ax_consensus, 'Consensus & Peer Dev',       'Value'),
            (self._adv_ax_slope,     'Regression Slope',           'Slope'),
            (self._adv_ax_r2,        'R\u00b2 (Goodness of Fit)',        'R\u00b2'),
            (self._adv_ax_flags,     'Detection Flags per Round',  'Count'),
        ]
        for ax, title, ylabel in titles:
            ax.set_title(title, color=HDR_FG, fontsize=10,
                         fontfamily=FONT_UI, pad=8, fontweight='semibold')
            ax.set_xlabel('Round', color=DIM, fontsize=8)
            ax.set_ylabel(ylabel,  color=DIM, fontsize=8)

        for i, name in enumerate(self.experiment_names):
            mdata = self._metrics_data.get(name, [])
            if not mdata:
                continue
            if name in self._hidden_exps:
                continue
            c     = LINE_COLOURS[i % len(LINE_COLOURS)]
            short = name[:22]
            rs    = [d['round'] for d in mdata]

            # Drift: mean line + std line + shaded band
            drift_mean = [d.get('drift_mean', 0) for d in mdata]
            drift_std  = [d.get('drift_std', 0)  for d in mdata]
            self._adv_ax_drift.plot(rs, drift_mean, color=c, lw=1.6,
                                     label=f'{short} mean',
                                     marker='o', markersize=2.5)
            self._adv_ax_drift.plot(rs, drift_std, color=c, lw=1.0,
                                     linestyle='--', alpha=0.7,
                                     label=f'{short} std')
            dm_arr = np.array(drift_mean)
            ds_arr = np.array(drift_std)
            self._adv_ax_drift.fill_between(rs, dm_arr - ds_arr,
                                             dm_arr + ds_arr,
                                             color=c, alpha=0.10)
            self._adv_hover_data.setdefault(
                id(self._adv_ax_drift), []).append(
                    (rs, drift_mean, f'{short} mean', c))

            # Consensus + peer deviation
            consensus = [d.get('consensus', 0)    for d in mdata]
            peer_dev  = [d.get('peer_dev_mean', 0) for d in mdata]
            self._adv_ax_consensus.plot(
                rs, consensus, color=c, lw=1.6,
                label=f'{short} consensus', marker='o', markersize=2.5)
            self._adv_ax_consensus.plot(
                rs, peer_dev, color=c, lw=1.0,
                linestyle='--', alpha=0.6, label=f'{short} peer dev')
            self._adv_hover_data.setdefault(
                id(self._adv_ax_consensus), []).append(
                    (rs, consensus, f'{short} consensus', c))

            # Slope (own chart)
            slopes = [d.get('slope', 0) for d in mdata]
            self._adv_ax_slope.plot(
                rs, slopes, color=c, lw=1.6,
                label=short, marker='o', markersize=2.5)
            self._adv_hover_data.setdefault(
                id(self._adv_ax_slope), []).append(
                    (rs, slopes, short, c))

            # R² (own chart)
            r_sqs = [d.get('r_squared', 0) for d in mdata]
            self._adv_ax_r2.plot(
                rs, r_sqs, color=c, lw=1.6,
                label=short, marker='o', markersize=2.5)
            self._adv_ax_r2.set_ylim(-0.05, 1.1)
            self._adv_hover_data.setdefault(
                id(self._adv_ax_r2), []).append(
                    (rs, r_sqs, short, c))

            # Detection flags count
            n_flagged = [d.get('n_flagged', 0) for d in mdata]
            self._adv_ax_flags.plot(
                rs, n_flagged, color=c, lw=1.6,
                label=short, marker='o', markersize=2.5)
            self._adv_hover_data.setdefault(
                id(self._adv_ax_flags), []).append(
                    (rs, n_flagged, short, c))

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER,
                          labelcolor=FG, loc='best', framealpha=0.92)

        self._adv_fig.tight_layout(pad=2.4)
        self._adv_canvas.draw_idle()

    # Utility
    def _fmt(self, val):
        if val is None:
            return '\u2013'
        try:
            return f'{float(val):.4f}'
        except (TypeError, ValueError):
            return str(val)
