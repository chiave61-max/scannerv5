def render_card(r):
    css = 'buy' if r['status']=='BUY' else 'sell' if r['status']=='SELL' else 'wait'
    sig_color = '#00ff88' if r['status']=='BUY' else '#ff3355' if r['status']=='SELL' else '#3a5070'
    sig_icon  = '▲ BUY' if r['status']=='BUY' else '▼ SELL' if r['status']=='SELL' else '⚪ ATTESA'
    score     = r['score_buy'] if r['status']=='BUY' else r['score_sell'] if r['status']=='SELL' else max(r['score_buy'], r['score_sell'])
    sc_class  = score_color(score)
    trend_ico = '📈' if r['uptrend'] else '📉'
    vol_str   = f"{r['vol_ratio']:.1f}x" if r['vol_ratio'] > 0 else 'N/A'

    levels = ''
    if r['status'] == 'BUY':
        rr = (r['tp_long'] - r['price']) / (r['price'] - r['sl_long']) if r['price'] > r['sl_long'] else 0
        levels = (
            "<div style='margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em'>"
            f"<span style='color:#00ff88'>TP: {r['tp_long']:.4f}</span> &nbsp;|&nbsp;"
            f"<span style='color:#ff3355'>SL: {r['sl_long']:.4f}</span> &nbsp;|&nbsp;"
            f"<span style='color:#ffcc00'>R/R: {rr:.1f}:1</span>"
            "</div>"
        )
    elif r['status'] == 'SELL':
        rr = (r['price'] - r['tp_short']) / (r['sl_short'] - r['price']) if r['sl_short'] > r['price'] else 0
        levels = (
            "<div style='margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em'>"
            f"<span style='color:#00ff88'>TP: {r['tp_short']:.4f}</span> &nbsp;|&nbsp;"
            f"<span style='color:#ff3355'>SL: {r['sl_short']:.4f}</span> &nbsp;|&nbsp;"
            f"<span style='color:#ffcc00'>R/R: {rr:.1f}:1</span>"
            "</div>"
        )

    st.markdown(f"""
    <div class='card {css}'>
        <div style='display:flex;justify-content:space-between;align-items:center'>
            <div>
                <b style='color:white;font-size:1.1em'>{r['name']}</b>
                <span style='color:#3a5070;font-size:0.75em;margin-left:8px'>{r['ticker']} · {r['cat'].upper()}</span>
            </div>
            <div style='text-align:right'>
                <b style='color:{sig_color};font-size:1em'>{sig_icon}</b><br>
                <span class='{sc_class}'>{score}/10</span>
            </div>
        </div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px'>
            <div><div class='label'>Prezzo</div><div class='value'>{r['price']:.4f}</div></div>
            <div><div class='label'>RSI</div><div class='value'>{r['rsi']}</div></div>
            <div><div class='label'>Volume</div><div class='value'>{vol_str}</div></div>
            <div><div class='label'>Trend</div><div class='value'>{trend_ico}</div></div>
        </div>
        {levels}
    </div>
    """, unsafe_allow_html=True)


