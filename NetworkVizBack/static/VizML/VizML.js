
function VizML(parentBlockId) {
    var workingDIV = d3.select("#" + parentBlockId).append("div")
        .attr("id", "VizML");

    // prepare the dashboard
    this.vizPanelSpan = workingDIV.append("span").attr("id", "dashBoardDiv");
    this.dashBoardSpan = workingDIV.append("span").attr("id", "vizPanelDiv");
}

VizML.prototype.initialViz = function(APIData) {
    console.log(APIData);
    // append node for sub svg 3 node in a row
    this.linkedData
    this.updateDashBoard(APIData);

}

VizML.prototype.updateDashBoard = function(APIData) {

}

VizML.prototype.addNode = function () {

}

VizML.prototype.addLink = function () {

}

VizML.prototype.removeNode = function () {

}

VizML.prototype.removeLink = function() {

}

export { VizML };