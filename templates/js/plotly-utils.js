function __interpolateX(p1, p2, y) {
    //(p1.y - p2.y) / (p1.y - y) = (p1.x - p2.x) / (p1.x - x)
    //p1.x - x = (p1.y - y) * (p1.x - p2.x) / (p1.y - p2.y)
    x = p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y)
    return x
}
function __findSegment(y, ys, low, high) {
    if (high - low === 2) return {low: low, high: high - 1};
    if (high - low === 1) return {low: low, high: high}

    let mid = Math.floor(( high + low ) / 2)
    console.log(mid, ys[mid], y, low, high)
    if (ys[mid] > y) {
        return __findSegment(y, ys, low, mid)
    }
    else {
        return __findSegment(y, ys, mid, high)
    }
    return {low: low, high: high}
}
function findSegment(y, ys) {
    let b = __findSegment(y, ys, 0, ys.length)
    if (b.high > y.length -1) {
        return { low: b.low -1, high: b.low }
    }
    return b
}
function interpolate(xs, ys, y) {
    let b = findSegment(y, ys)
    return __interpolateX({x: xs[b.low], y: ys[b.low]}, {x: xs[b.high], y: ys[b.high]}, y)
}
function transformLabels(selector, xs, ys) {
    //let lblElems = document.querySelectorAll('.y2tick text')
    let lblElems = document.querySelectorAll(selector)
    for (let elem of lblElems) {
        let y = elem.__data__.x
        elem.innerHTML = interpolate(xs, ys, y).toFixed(1)
    }
}
async function loadPlot(elem,path) {
    const response = await fetch(urlParams.get('plot') || '/linhtinh.json');
    const chart = await response.json();
    let HOLDER = elem;
    if (typeof elem === 'string') {
        HOLDER = document.getElementById(elem);
    }
    await Plotly.newPlot( HOLDER, chart.data, chart.layout )
    
    let tvdss = HOLDER.data.find(function(item) {
        return item.name === 'TVDSS'
    })
    let xs = tvdss.x._inputArray
    let ys = tvdss.y._inputArray
    transformLabels('.y2tick text', xs, ys)
    HOLDER.on('plotly_relayout', function(eventData) {
        console.log('eventData', eventData);
        transformLabels('.y2tick text', xs, ys)
    });
    return HOLDER
}

const urlParams = new URLSearchParams(window.location.search);
plot = loadPlot('holder', urlParams.get('plot') || '/linhtinh.json');
