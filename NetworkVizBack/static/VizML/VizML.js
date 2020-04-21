
function VizML(parentBlockId) {
    var workingDIV = d3.select("#" + parentBlockId).append("div")
        .attr("id", "VizML");

    this.VizTooltip = workingDIV.append("div")
        .attr("class", "VizTooltip")
        .style("opacity", 0);

    var linkedVizML = this;

    /* data structure for graph */
    this.nodeId = 0;
    this.linkId = 0;
    this.blockId = 0;
    this.nodeRecorder = new Map();
    this.linkRecorder = new Map();
    this.blockRecorder = new Map();

    /* prepare the dashboard */
    this.dashBoardDivWidth = 25;
    this.dashBoardDivHeight = 87;
    this.dashBoardDiv = workingDIV.append("div")
        .attr("class", "dashBoardDiv")
        .style("width", this.dashBoardDivWidth + "vw")
        .style("height", this.dashBoardDivHeight + "vh");
    this.vizPanelDivWidth = 70;
    this.vizPanelDivHeight = 87;
    this.vizPanelDiv = workingDIV.append("div")
        .attr("class", "vizPanelDiv")
        .style("width", this.vizPanelDivWidth + "vw")
        .style("height", this.vizPanelDivHeight + "vh");
    this.vizPanelSVG = this.vizPanelDiv.append("svg").attr("class", "vizPanelSVG")
        .attr("width", "100%")
        .attr("height", "100%");
    /* append bottom for interaction - group, generate code */
    this.buttonDiv = workingDIV.append("div").attr("class", "buttonDiv");
    this.buttonDiv.append("div").attr("class", "buttonHolder")
        .append("button").attr("type", "button")
        .attr("class","btn btn-info groupButton")
        .text(function () {
            this.linkedVizML = linkedVizML;
            return "Group Block";
        });  // add click event handler
    this.buttonDiv.append("div").attr("class", "buttonHolder")
        .append("button").attr("type", "button")
        .attr("class","btn btn-warning generateButton")
        .text(function () {
            this.linkedVizML = linkedVizML;
            return "Generate Code";
        }); // add click event handler
}

VizML.prototype.initialViz = function(APIData) {
    // append node for sub svg 2 node in a row
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

VizML.prototype.getBlockId = function() {
    var generateId = this.blockId;
    this.blockId++;
    return generateId;
};

VizML.prototype.updateDashBoard = function(APIData) {
    /* add node */
    var linkedVizML = this;
    var nodePerRow = 2;
    var totalTypeCount = 0;
    var totalLayerCount = 0;
    for (var nodeType in APIData) {
        totalTypeCount++;
        totalLayerCount += (Math.ceil(APIData[nodeType].length/nodePerRow));
    }

    var dashBoardWidth = this.dashBoardDivWidth;
    var dashBoardHeight = this.dashBoardDivHeight;

    var counter = 0;
    var badgeType = ["badge-primary", "badge-secondary", "badge-success", "badge-warning", "badge-danger", "badge-light", "badge-dark"];
    var nodeColor = ["#8dd3c7", "#bc80bd", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#bebada"];
    for (var nodeType in APIData) {
        var sectionRowCount = Math.ceil(APIData[nodeType].length/nodePerRow);
        var sectionHeight = sectionRowCount/totalLayerCount * (dashBoardHeight - totalTypeCount * 3.7) + 3;
        var innerDiv = this.dashBoardDiv.append("div")
            .attr("class", "nodeTypeDashDiv")
            .style("width", (dashBoardWidth - 1) + "vw")
            .style("height", sectionHeight + "vh");
        var badgeDiv = innerDiv.append("div").attr("class", "nodeTypeTitleDiv");
        badgeDiv.append("span").attr("class", "badge " + badgeType[counter] + " nodeTypeTitle")
            .text(nodeType);
        var SVGDivHeight = document.documentElement.clientHeight * sectionHeight/100 - parseInt(badgeDiv.style("height"));
        var innerSVG = innerDiv.append("div").attr("class", "nodeTypeSVGDiv")
            .style("width", (dashBoardWidth - 1) + "vw")
            .style("height", SVGDivHeight + "px")
            .append("svg").attr("class", "nodeTypeSVG")
            .attr("width", "100%")
            .attr("height", "100%");

        /* get the actual svg height and width */
        var svgWidth = parseInt(innerSVG.style("width"), 10);
        var svgHeight = parseInt(innerSVG.style("height"), 10);
        var APINode = innerSVG.selectAll(".APINode").data(APIData[nodeType])
            .enter().append("g").attr("class", "APINode").attr("transform", function (d, i) {
                this.linkedVizML = linkedVizML;
                d["color"] = nodeColor[counter];
                return "translate(" + i%nodePerRow * (svgWidth/nodePerRow) + ", " + Math.floor(i/nodePerRow)/sectionRowCount * svgHeight + ")";
            });
        APINode.selectAll("circle").data(d => [d]).enter()
            .append("circle")
            .attr("class", "APINodeDot")
            .attr("fill", d=>d["color"])
            .attr("cx", 18)
            .attr("cy", 12)
            .attr("r", 10);
        APINode.selectAll("text").data(d => [d]).enter()
            .append("text")
            .attr("class", "APINodeText")
            .attr("transform", "translate(38, 18)")
            .text(d => d.node);
        counter++;
    }

    /* event register */
    this.dashBoardDiv.selectAll(".APINode")
        .on("mouseenter", APINodeEnter)
        .on("mouseout", APINodeOut)
        .on("click", APINodeClick);
};

VizML.prototype.addNode = function (NodeInfo) {
    var generatedNodeInfo;
    if (NodeInfo.hasOwnProperty("id")) {
        generatedNodeInfo = NodeInfo;
    } else {
        generatedNodeInfo = JSON.parse(JSON.stringify(NodeInfo)); // return a deep copy of the node
        generatedNodeInfo["id"] = this.getNodeId();
        generatedNodeInfo["ports"] = new Set(generatedNodeInfo["ports"]);
    }
    if (!generatedNodeInfo.hasOwnProperty("position"))
        generatedNodeInfo["position"] = {"x": 400, "y": 200};
    // add to node map
    console.log(generatedNodeInfo);
    var generatedNode = this.vizPanelSVG.append("g").attr("class", "vizNode")
        .datum(generatedNodeInfo)
        .attr("transform", "translate(" + generatedNodeInfo["position"]["x"] + ", " + generatedNodeInfo["position"]["y"] + ")");
    generatedNode.selectAll(".vizNodePort").data(generatedNodeInfo["ports"]).enter().append("circle")
        .attr("class", "vizNodePort").attr("cx", function (d) {

        })
    generatedNode.selectAll(".vizNodeRect").data(d => [d]).enter().append("rect")
        .attr("class", "vizNodeRect")

    this.removeNode(nodeID);

    //this.nodeRecorder.set(generatedNodeInfo["id"])

    // update in the global map (maintain the link) - blockid, linkid, remove original link, add new link

};

// double click to remove node
VizML.prototype.removeNode = function (nodeID) {
    if (this.nodeRecorder.has(nodeID)) {
        /* remove the node and all related links and block relation */
    }
};

VizML.prototype.addLink = function (LinkInfo) {

};



// double click to remove link
VizML.prototype.removeLink = function() {

};

VizML.prototype.addBlock = function (BlockInfo) {
    //alert("Node test");
}

// double click to remove block
VizML.prototype.removeBlock = function() {
    /* for all node in block, recreate the node */
    /* for all link for nodes, recreate the link */
}

function APINodeEnter(e) {
    var linkedVizML = this.linkedVizML;
    var VizTooltip = linkedVizML.VizTooltip;
    var linkedData = d3.select(this).datum();
    if (linkedData !== undefined) {
        VizTooltip.transition()
        .style("opacity", .9);
        VizTooltip.html(
            '<h4>' + linkedData["node"] + '</h4>' +
            '<p>' + "Decription:<br>" + linkedData["description"] + '<br><br>' +
            'Ports:  ' + linkedData["ports"] + '</p>'
        ).style("left", (d3.event.pageX + 10) + "px")
            .style("top", (d3.event.pageY - 30) + "px");
    }
}

function APINodeOut(e) {
    var linkedVizML = this.linkedVizML;
    var VizTooltip = linkedVizML.VizTooltip;
    VizTooltip.html("")
        .style("opacity", 0);
}

function APINodeClick(e) {
    var linkedVizML = this.linkedVizML;
    var linkedData = d3.select(this).datum();
    linkedVizML.addNode(linkedData);
}

export { VizML };