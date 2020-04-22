const line = d3.line()
  .x(d=>d.x)
  .y(d=>d.y)
  .curve(d3.curveCatmullRom.alpha(.5))

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
    this.nodeRecorder = new Map();
    this.linkRecorder = new Map();
    this.blockRecorder = new Map();
    this.portIllustration = ["Main Input", "Sub Input", "Meta Info", "Main Output", "Sub Output"];
    this.portColorMapping = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"];

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
    this.buttonDiv.append("div").attr("class", "buttonHolder")
        .append("button").attr("type", "button")
        .attr("class","btn btn-danger uploadButton")
        .text(function () {
            this.linkedVizML = linkedVizML;
            return "Import Model";
        }); // add click event handler
}

VizML.prototype.initialViz = function(APIData) {
    // append node for sub svg 2 node in a row
    var linkedVizML = this;
    this.linkedData = APIData;
    this.updateDashBoard(this.linkedData);
    var svgHeight = parseInt(this.vizPanelSVG.style("height"), 10);
    console.log(svgHeight);
    var portLegenLabel = this.vizPanelSVG.append("g").attr("class", "portLegend")
        .attr("transform", "translate(10, " + (svgHeight - 40) + ")")
        .selectAll(".portLegendLabel")
        .data([0, 1, 2, 3, 4]).enter().append("g").attr("class", "portLegendLabel").attr("transform", function (d) {
            return "translate(" + d * 160 + " , 0)";
        });
    portLegenLabel.selectAll(".portLegendLabelDot").data(d=>[d]).enter().append("circle")
        .attr("class", "portLegendLabelDot")
        .attr("cx", 15).attr("cy", 15).attr("r", 10)
        .attr("fill", d=>linkedVizML.portColorMapping[d]);
    portLegenLabel.selectAll(".portLegendLabelText").data(d=>[d]).enter().append("text").attr("class", "portLegendLabelText")
        .attr("transform", "translate(30, 22)")
        .text(d=>linkedVizML.portIllustration[d]);
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
    return this.getNodeId();
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
    var linkedVizML = this;

    var generatedNodeInfo;
    if (NodeInfo.hasOwnProperty("id")) {
        generatedNodeInfo = NodeInfo;
    } else {
        generatedNodeInfo = JSON.parse(JSON.stringify(NodeInfo)); // return a deep copy of the node
        generatedNodeInfo["id"] = this.getNodeId();
        generatedNodeInfo["ports"] = new Set(generatedNodeInfo["ports"]);
        generatedNodeInfo["portMap"] = new Map();
        generatedNodeInfo["assignedBlock"] = -1;
    }
    if (!generatedNodeInfo.hasOwnProperty("position"))
        generatedNodeInfo["position"] = {"x": 400, "y": 200};
    // add to node map
    console.log(generatedNodeInfo);
    var generatedNode = this.vizPanelSVG.append("g").datum(generatedNodeInfo)
        .attr("class", "vizNode")
        .attr("transform", function () {
            this.linkedVizML = linkedVizML;
            return "translate(" + generatedNodeInfo["position"]["x"] + ", " + generatedNodeInfo["position"]["y"] + ")";
        });

    /* port generation */
    var portRadius = 10;
    var rectWidth = 180;
    var rectHeight = 35;
    var rectPositionX = generatedNodeInfo["ports"].has(3)? 2*portRadius + 1:10;
    var rectPositionY = 10;

    if (generatedNodeInfo["ports"].has(4) || generatedNodeInfo["ports"].has(5)) {
        let linkedData = [4];
        if (generatedNodeInfo["ports"].has(4) && generatedNodeInfo["ports"].has(5))
            linkedData = [4, 5];

        generatedNode.selectAll(".vizNodePort").data(linkedData).enter()
            .append("circle").attr("class", "vizNodePort")
            .attr("cx", function (d, i) {
                generatedNodeInfo["portMap"].set(d, d3.select(this));
                this.linkedVizML = linkedVizML;
                this.linkedNodeId = generatedNodeInfo["id"];
                return rectPositionX + rectWidth/(2 * linkedData.length) * (1 + 2 * i);
            })
            .attr("cy", portRadius)
            .attr("r", portRadius)
            .attr("fill", d=>linkedVizML.portColorMapping[d-1]);
        rectPositionY = 2*portRadius + 1;
    }
    if (generatedNodeInfo["ports"].has(3)) {
        generatedNode.append("circle").attr("class", "vizNodePort")
            .attr("cx", function (d, i) {
                d3.select(this).datum(3);
                generatedNodeInfo["portMap"].set(3, d3.select(this));
                this.linkedVizML = linkedVizML;
                this.linkedNodeId = generatedNodeInfo["id"];
                return portRadius;
            })
            .attr("cy", rectPositionY + rectHeight/2)
            .attr("r", portRadius)
            .attr("fill", d=>linkedVizML.portColorMapping[d-1]);
    }

    generatedNode.append("rect").attr("class", "vizNodeRect")
        .attr("x", rectPositionX).attr("y", rectPositionY)
        .attr("width", rectWidth)
        .attr("height", rectHeight)
        .attr("fill", generatedNodeInfo["color"]);
    generatedNode.append("text").attr("class", "vizNodeText")
        .attr("transform", "translate(" + (rectPositionX + rectWidth/2) + " , " + (rectPositionY + rectHeight/2 + 6) + ")")
        .text(generatedNodeInfo["node"]);

    if (generatedNodeInfo["ports"].has(1) || generatedNodeInfo["ports"].has(2)) {
        let linkedData = [1];
        if (generatedNodeInfo["ports"].has(1) && generatedNodeInfo["ports"].has(2))
            linkedData = [1, 2];
        let counter = 0;
        linkedData.forEach(function (d) {
            generatedNode.append("circle").attr("class", "vizNodePort")
            .attr("cx", function () {
                d3.select(this).datum(d);
                generatedNodeInfo["portMap"].set(d, d3.select(this));
                this.linkedVizML = linkedVizML;
                this.linkedNodeId = generatedNodeInfo["id"];
                return rectPositionX + rectWidth/(2 * linkedData.length) * (1 + 2 * counter);
            })
            .attr("cy", rectPositionY + rectHeight + portRadius + 1)
            .attr("r", portRadius)
            .attr("fill", d=>linkedVizML.portColorMapping[d-1]);
            counter++;
        })
    }

    /* remove old node and update node */
    if (linkedVizML.nodeRecorder.has(generatedNodeInfo["id"]))
        d3.select(linkedVizML.nodeRecorder.get(generatedNode["id"])).remove();
    linkedVizML.nodeRecorder.set(generatedNodeInfo["id"], generatedNode)

    /* handle with link */
    if (generatedNodeInfo.hasOwnProperty("links")) {
        /* remove old links, and regenerate the links */
        generatedNodeInfo["links"].forEach(function (LinkID) {
            linkedVizML.removeLink(LinkID);
        });
        linkedVizML.removeLink();
    } else {
        generatedNodeInfo["links"] = new Set(); // link id as key - will update the g element
    }

    /* drag event listener */
    generatedNode.call(d3.drag()
        .on("start", dragGenerateNodeStart)
        .on("drag", dragGenerateNode)
        .on("end", dragGenerateNodeEnd));

    /* double click event listener */
    generatedNode.on("dbclick", dbclickGeneratedNode);
};

// double click to remove node
VizML.prototype.removeNode = function (nodeID) {
    var linkedVizML = this;
    var generatedNodeInfo = linkedVizML.nodeRecorder.get(nodeID);
    if (linkedVizML.nodeRecorder.has(nodeID)) {
        /* remove the block relation */
        if (generatedNodeInfo["assignedBlock"] !== -1) {
            let linkedBlockInfo = linkedVizML.blockRecorder.get(generatedNodeInfo["assignedBlock"]).datum();
            linkedBlockInfo["nodes"].delete(nodeID);
            if (linkedBlockInfo["nodes"].size === 0)
                this.removeBlock(linkedBlockInfo["id"]);
        }
        /* remove the link relation */
        let nodeLinks = new Set();
        generatedNodeInfo["links"].forEach(function (linkID) {
            let newLinkID = linkedVizML.addLink(linkedVizML.linkRecorder.get(linkID).datum());
            nodeLinks.add(newLinkID);
            linkedVizML.removeLink(linkID);
        });
        /* delete the node */
        linkedVizML.nodeRecorder.get(nodeID).remove();
    }
};

VizML.prototype.addLink = function (LinkInfo) {
    var generatedLinkInfo = JSON.parse(JSON.stringify(LinkInfo)); // return a deep copy of the node
    generatedLinkInfo["id"] = this.getLinkId();
    /* add link */

    return generatedLinkInfo["id"]
};

// double click to remove link
VizML.prototype.removeLink = function(linkID) {
    if (this.linkRecorder.has(linkID))
        this.linkRecorder.get(linkID).remove();
};

VizML.prototype.addBlock = function (BlockInfo) {
    //alert("Node test");
}

// double click to remove block
VizML.prototype.removeBlock = function() {
    /* just remove is ok */
}

VizML.prototype.generateCode = function () {
    var linkedData = this.linkedData;
    console.log(linkedData);
    $.ajax({
			url: '/ModelGeneration',
			data: JSON.stringify(linkedData),
			type: 'POST',
            contentType: "application/json; charset=utf-8",
			success: function(response){
				console.log(response);
			},
			error: function(error){
				alert(error);
			}
		});
}

/* event handler for api node */
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


/* event handler for viz node */
function dbclickGeneratedNode() {
    var linkedVizML = this.linkedVizML;
    var generatedNodeInfo = d3.select(this).datum();
    console.log(linkedVizML);
    console.log(generatedNodeInfo);
    linkedVizML.removeNode(generatedNodeInfo["id"]);
}

function dragGenerateNodeStart() {
    /* remove the generated link */
}

function dragGenerateNode() {
    /* update the position of node */
    d3.select(this)
        .attr("transform", "translate(" + d3.event.x + "," + d3.event.y + ")");
}

function dragGenerateNodeEnd() {
    var linkedData = d3.select(this).datum();
    linkedData["position"]["x"] = d3.event.x;
    linkedData["position"]["y"] = d3.event.y;
    d3.select(this)
        .attr("transform", "translate(" + linkedData["position"]["x"] + ", " + linkedData["position"]["y"] + ")");
    var linkedVizML = this.linkedVizML;

    /* recreate link */
    linkedVizML.vizPanelSVG.append('path')
        .datum([{x: 100, y: 200}, {x: linkedData["position"]["x"], y: linkedData["position"]["y"]}])
        .attr('d', line)
        .style("stroke", "red")
        .style("stroke-width", 2);
    console.log(linkedData["portMap"]);
    //console.log(linkedVizML);
    //console.log(d3.select(this).datum())
    //console.log(linkedData)
}

/* event handler for viz link */
function dbclickGeneratedLink() {

}

/* event handler for viz block */
function dbclickGeneratedBlock() {

}

function

export { VizML };