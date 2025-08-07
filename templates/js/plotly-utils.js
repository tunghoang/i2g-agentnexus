async function loadPlot(elem,path) {
    const response = await fetch(urlParams.get('plot') || '/linhtinh.json');
    const chart = await response.json();
    let HOLDER = elem;
    if (typeof elem === 'string') {
        HOLDER = document.getElementById(elem);
    }
    Plotly.newPlot( HOLDER, chart.data, chart.layout )
    return HOLDER
}

const urlParams = new URLSearchParams(window.location.search);
plot = loadPlot('holder', urlParams.get('plot') || '/linhtinh.json');
plot.on('plotly_relayout', function(eventData) {
    console.log('eventData', eventData);
});
