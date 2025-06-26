def layout(direction='horizontal', children = [], style=""):
    return f'''<div class='layout-{direction}' style='{style}'>
    {''.join(children)}
</div>'''
def stack(children=[], style=""):
    return f'''<div class='stack' style='{style}'>{''.join(children)}</div>'''
