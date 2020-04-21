
function VizML(parentBlockId) {
    var workingDIV = d3.select("#" + parentBlockId).append("div")
        .attr("id", "VizML");

    this.nodeId = 0;
    this.linkId = 0;
    //this.nodeRecorder;
    //this.linkRecorder;
    //this.blockRecorder;
    /* prepare the dashboard */
    this.dashBoardDivWidth = "350px";
    this.dashBoardDivHeight = "900px";
    this.dashBoardDiv = workingDIV.append("div")
        .attr("class", "dashBoardDiv")
        .style("width", this.dashBoardDivWidth)
        .style("height", this.dashBoardDivHeight);
    this.vizPanelDivWidth = "1400px";
    this.vizPanelDivHeight = "900px";
    this.vizPanelDiv = workingDIV.append("div")
        .attr("class", "vizPanelDiv")
        .style("width", this.vizPanelDivWidth)
        .style("height", this.vizPanelDivHeight);
    /* append bottom for interaction - group, generate code */
    this.buttonDiv = workingDIV.append("div").attr("class", "buttonDiv");
    this.buttonDiv.append("div").attr("class", "buttonHolder")
        .append("button").attr("type", "button")
        .attr("class","btn btn-info groupButton")
        .text("Group Block");
    this.buttonDiv.append("div").attr("class", "buttonHolder")
        .append("button").attr("type", "button")
        .attr("class","btn btn-warning generateButton")
        .text("Generate Code");
}

VizML.prototype.initialViz = function(APIData) {
    console.log(APIData);
    // append node for sub svg 3 node in a row
    this.linkedData = APIData;
    this.updateDashBoard(this.linkedData);
};

VizML.prototype.getNodeId = function() {
    var generateId = this.nodeId;
    this.nodeId++;
    return generateId;
};

VizML.prototype.getLinkId = function() {
    var generateId = this.linkId;
    this.linkId++;
    return generateId;
};

VizML.prototype.updateDashBoard = function(APIData) {
    /* add node */


    /* event register */
};

VizML.prototype.addNode = function (NodeInfo) {
    var generatedNode = JSON.parse(JSON.stringify(NodeInfo)); // return a deep copy of the node
    if (!generatedNode.hasOwnProperty("id"))
        generatedNode["id"] = this.getNodeId();
    if (!generateNode.hasOwnProperty("position"))
        generatedNode["position"] = {"x": 200, "y": 200};

    // add to node map

    // update in the global map (maintain the link)

};

VizML.prototype.addLink = function () {

};

VizML.prototype.removeNode = function () {

};

VizML.prototype.removeLink = function() {

};

export { VizML };