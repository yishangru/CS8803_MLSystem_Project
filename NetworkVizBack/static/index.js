import { VizML }  from "./VizML/VizML.js"
var PyTorchAPI = "./static/API/PyTorch/VizAPI.json";
var PyTorchViz = new VizML("main");

// load API data
Promise.all([
    d3.json(PyTorchAPI)
]).then(function (data) {
    // parse data for loading
    var PyTorchNode = d3.nest()
        .key(function(d) { return d.type})
        .object(data[0]);
    PyTorchViz.initialViz(PyTorchNode);
});