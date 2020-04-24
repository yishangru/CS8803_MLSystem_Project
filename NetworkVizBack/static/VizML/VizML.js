const Vizlink = d3.linkVertical()
    .source(d=> [d["source"]["x"], d["source"]["y"]])
    .target(d=> [d["target"]["x"], d["target"]["y"]]);


function VizML(parentBlockId) {
    var workingDIV = d3.select("#" + parentBlockId).append("div")
        .attr("id", "VizML");

    this.VizTooltip = workingDIV.append("div")
        .attr("class", "VizTooltip")
        .style("opacity", 0);

    this.VizParamEnter = workingDIV.append("div")
        .attr("class", "VizParamEnter")
        .style("opacity", 0);

    var linkedVizML = this;

    /* data structure for graph */
    this.nodeId = 0;
    this.linkId = 0;
    this.nodeRecorder = new Map();
    this.linkRecorder = new Map();
    this.blockRecorder = new Map();
    this.portIllustration = ["Main Input (require gradient)", "Sub Input (loss)", "Meta Info", "Main Output", "Sub Output (label)"];
    this.portColorMapping = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"];

    /* recorder current selection */
    this.currentSelection = {source: {nodeID:null, port:null}, target: {nodeID:null, port:null}}; // for port link
    this.currentLayerSelection = new Set();  // for block generation

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
    this.vizPanelSVG.append('defs')
        .append('marker')
        .attr('id', 'linkArrow')
        .attr('refX', 0)
        .attr('refY', 2)
        .attr('markerWidth', 4)
        .attr('markerHeight', 4)
        .attr('orient', 'auto-start-reverse')
        .append('polygon')
        .attr('points', '0 0, 4 2, 0 4')
        .attr('fill', '#737373')
        .style('stroke', '#bdbdbd')
        .style('stroke-width', 0.5);

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
    var portLegenLabel = this.vizPanelSVG.append("g").attr("class", "portLegend")
        .attr("transform", "translate(10," + (svgHeight - 40) + ")")
        .selectAll(".portLegendLabel")
        .data([0, 1, 2, 3, 4]).enter().append("g").attr("class", "portLegendLabel").attr("transform", function (d) {
            if (d <= 1) {
                return "translate(" + d * 380 + ",0)";
            } else if (d === 2) {
                return "translate(" + (380 + (d - 1) * 220) + ",0)";
            } else {
                return "translate(" + (600 + (d - 2) * 180) + ",0)";
            }
        });
    portLegenLabel.selectAll(".portLegendLabelDot").data(d=>[d]).enter().append("circle")
        .attr("class", "portLegendLabelDot")
        .attr("cx", 15).attr("cy", 15).attr("r", 10)
        .attr("fill", d=>linkedVizML.portColorMapping[d]);
    portLegenLabel.selectAll(".portLegendLabelText").data(d=>[d]).enter().append("text").attr("class", "portLegendLabelText")
        .attr("transform", "translate(30,22)")
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
                return "translate(" + i%nodePerRow * (svgWidth/nodePerRow) + "," + Math.floor(i/nodePerRow)/sectionRowCount * svgHeight + ")";
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
            .attr("transform", "translate(38,18)")
            .text(d => d.node);
        counter++;
    }

    /* event register */
    this.dashBoardDiv.selectAll(".APINode")
        .on("mouseenter", APINodeEnter)
        .on("mouseout", APINodeOut)
        .on("click", APINodeClick);
};

// click API to add node
VizML.prototype.addNode = function (NodeInfo) {
    var linkedVizML = this;

    var generatedNodeInfo;
    if (NodeInfo.hasOwnProperty("id")) {
        generatedNodeInfo = NodeInfo;
        generatedNodeInfo["portMap"].clear();
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
    var generatedNode = this.vizPanelSVG.append("g").datum(generatedNodeInfo)
        .attr("class", "vizNode")
        .attr("transform", function () {
            this.linkedVizML = linkedVizML;
            return "translate(" + generatedNodeInfo["position"]["x"] + "," + generatedNodeInfo["position"]["y"] + ")";
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

    var vizNodePanel = generatedNode.append("g").attr("class", "vizNodePanel")
        .attr("transform", function () {
            this.linkedVizML = linkedVizML;
            this.linkedNodeId = generatedNodeInfo["id"];
            return "translate(" + rectPositionX + "," + rectPositionY + ")";
        });
    vizNodePanel.append("rect").attr("class", "vizNodeRect")
        .attr("x", 0).attr("y", 0)
        .attr("width", rectWidth)
        .attr("height", rectHeight)
        .attr("fill", generatedNodeInfo["color"]);
    vizNodePanel.append("text").attr("class", "vizNodeText")
        .attr("transform", "translate(" + rectWidth/2 + "," + (rectHeight/2 + 6) + ")")
        .text(generatedNodeInfo["node"]);
    generatedNode.append("text").attr("class", "vizNodeParam")
        .attr("transform", function () {
            this.linkedVizML = linkedVizML;
            this.linkedNodeId = generatedNodeInfo["id"];
            return "translate(" + (rectPositionX + rectWidth + 5) + "," + (rectPositionY + rectHeight/2 + 6) + ")"
        })
        .text("▼").on("click", clickParam); // ▼▲

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
    if (linkedVizML.nodeRecorder.has(generatedNodeInfo["id"])) {
        /* restore parameter entering */
        let currentNode = linkedVizML.nodeRecorder.get(generatedNode["id"]);

        currentNode.remove();
    }
    linkedVizML.nodeRecorder.set(generatedNodeInfo["id"], generatedNode)

    /* handle with link */
    if (generatedNodeInfo.hasOwnProperty("links")) {
        /* remove old links, and regenerate the links, update link recorder */
        generatedNodeInfo["links"].forEach(function (linkID) {
            linkedVizML.addLink(linkedVizML.linkRecorder.get(linkID).datum());
        });
    } else {
        generatedNodeInfo["links"] = new Set(); // link id as key - will update the g element
    }

    /* event handler for port */
    generatedNode.selectAll(".vizNodePort").on("click", clickPort);
    /* event handler for layer */
    generatedNode.select(".vizNodePanel").on("click", clickPanel);

    /* drag event listener */
    generatedNode.call(d3.drag()
        .on("start", dragGenerateNodeStart)
        .on("drag", dragGenerateNode)
        .on("end", dragGenerateNodeEnd));

    /* double click event listener */
    generatedNode.on("dblclick", dbclickGeneratedNode);
};

// double click to remove node
VizML.prototype.removeNode = function (nodeID) {
    var linkedVizML = this;
    var generatedNodeInfo = linkedVizML.nodeRecorder.get(nodeID).datum();

    /* remove the data in selection */
    if (linkedVizML.currentLayerSelection.has(nodeID))
        linkedVizML.currentLayerSelection.delete(nodeID);
    if (linkedVizML.currentSelection.source.nodeId === nodeID) {
        linkedVizML.currentSelection.source.nodeId = undefined;
        linkedVizML.currentSelection.source.port = undefined;
    }

    if (linkedVizML.nodeRecorder.has(nodeID)) {
        /* remove the block relation */
        if (generatedNodeInfo["assignedBlock"] !== -1) {
            let linkedBlockInfo = linkedVizML.blockRecorder.get(generatedNodeInfo["assignedBlock"]).datum();
            linkedBlockInfo["nodes"].delete(nodeID);
            if (linkedBlockInfo["nodes"].size === 0)
                this.removeBlock(linkedBlockInfo["id"]);
        }
        /* remove the link relation */
        generatedNodeInfo["links"].forEach(function (linkID) {
            linkedVizML.removeLink(linkID, generatedNodeInfo["id"]);
        });
        generatedNodeInfo["links"].clear();
        /* restore the param enter */


        /* delete the node */
        linkedVizML.nodeRecorder.get(nodeID).remove();
        linkedVizML.nodeRecorder.delete(nodeID);
    }
};

// add link between two selected nodes and ports
VizML.prototype.addLink = function (LinkInfo) {
    var generatedLinkInfo;
    var linkedVizML = this;
    var sourceNodeInfo = linkedVizML.nodeRecorder.get(LinkInfo["source"]["nodeID"]).datum();
    var targetNodeInfo = linkedVizML.nodeRecorder.get(LinkInfo["target"]["nodeID"]).datum();
    var sourceNodePort = LinkInfo["source"]["port"];
    var targetNodePort = LinkInfo["target"]["port"];

    /* generate the link id */
    if (LinkInfo.hasOwnProperty("id")) {
        generatedLinkInfo = LinkInfo;  // recreate the link
    } else {
        /* check whether the link exist */
        for (let linkID of sourceNodeInfo["links"].values()) {
            let linkInfo = linkedVizML.linkRecorder.get(linkID).datum();
            if (linkInfo["source"]["nodeID"] === sourceNodeInfo["id"] &&
                linkInfo["target"]["nodeID"] === targetNodeInfo["id"] &&
                linkInfo["source"]["port"] === sourceNodePort &&
                linkInfo["target"]["port"] === targetNodePort) {
                return;
            }
        }
        /* link is not exist */
        generatedLinkInfo = JSON.parse(JSON.stringify(LinkInfo));  // return a deep copy of the link
        generatedLinkInfo["id"] = this.getLinkId();

        /* add to target and source node links */
        sourceNodeInfo["links"].add(generatedLinkInfo["id"]);
        targetNodeInfo["links"].add(generatedLinkInfo["id"]);
    }

    /* get link coordination */
    var sourcePort = sourceNodeInfo["portMap"].get(sourceNodePort);
    sourcePort.attr("stroke", "black").attr("stroke-width", 4);
    var sourcePortCenter = {
        x: parseInt(sourcePort.attr("cx"), 10),
        y: parseInt(sourcePort.attr("cy"), 10)
    };
    var targetPort = targetNodeInfo["portMap"].get(targetNodePort);
    targetPort.attr("stroke", "black").attr("stroke-width", 4);
    var targetPortCenter = {
        x: parseInt(targetPort.attr("cx"), 10),
        y: parseInt(targetPort.attr("cy"), 10)
    };
    var portRadius = parseInt(sourcePort.attr("r"), 10);
    var startPoint = {x: sourceNodeInfo["position"]["x"] + sourcePortCenter["x"], y: sourceNodeInfo["position"]["y"] + sourcePortCenter["y"]};
    var targetPoint = {x: targetNodeInfo["position"]["x"] + targetPortCenter["x"], y: targetNodeInfo["position"]["y"] + targetPortCenter["y"]};

    if (targetPoint.y - 2*portRadius >= startPoint.y) {
        targetPoint["y"] = targetPoint["y"] - 2*portRadius;
    } else if (targetPoint.y + 2*portRadius <= startPoint.y) {
        targetPoint["y"] = targetPoint["y"] + 2*portRadius;
    }
    /* add link to viz */
    var generatedLink = linkedVizML.vizPanelSVG.append("path")
        .attr("class", "vizLink")
        .attr("d", Vizlink({source: startPoint, target: targetPoint}))
        .attr("marker-end", "url(#linkArrow)")
        .attr('stroke', function () {
            this.linkedVizML = linkedVizML;
            return generatedLinkInfo["color"];
        })
        .attr('fill', 'none')
        .attr("stroke-width", 6);
    generatedLink.datum(generatedLinkInfo);

    /* remove old node and update node */
    if (linkedVizML.linkRecorder.has(generatedLinkInfo["id"]))
        linkedVizML.linkRecorder.get(generatedLinkInfo["id"]).remove();
    linkedVizML.linkRecorder.set(generatedLinkInfo["id"], generatedLink);

    /* add dbclick event handler */
    generatedLink.on("dblclick", dbclickGeneratedLink);
};

// double click to remove link
VizML.prototype.removeLink = function(linkID, nodeID=null) {
    if (this.linkRecorder.has(linkID)) {
        let linkInfo = this.linkRecorder.get(linkID).datum();
        let sourceInfo = this.nodeRecorder.get(linkInfo["source"]["nodeID"]).datum();
        let targetInfo = this.nodeRecorder.get(linkInfo["target"]["nodeID"]).datum();
        sourceInfo["portMap"].get(linkInfo["source"]["port"]).attr("stroke", "none").attr("stroke-width", 0);
        targetInfo["portMap"].get(linkInfo["target"]["port"]).attr("stroke", "none").attr("stroke-width", 0);

        if (nodeID !== null) {
            if (linkInfo["source"]["nodeID"] === nodeID) {
                targetInfo["links"].delete(linkID);
            } else if (linkInfo["target"]["nodeID"] === nodeID) {
                sourceInfo["links"].delete(linkID);
            }
        } else {
            sourceInfo["links"].delete(linkID);
            targetInfo["links"].delete(linkID);
        }

        this.linkRecorder.get(linkID).remove();
        this.linkRecorder.delete(linkID);
    }
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


VizML.prototype.importModel = function () {

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
            .style("top", (d3.event.pageY - 30) + "px")
            .style("width", "200px")
            .style("padding", "2px 5px 0px 5px");
    }
}

function APINodeOut(e) {
    var linkedVizML = this.linkedVizML;
    var VizTooltip = linkedVizML.VizTooltip;
    VizTooltip.html("")
        .style("width", 0)
        .style("padding", 0)
        .style("opacity", 0);
}

function APINodeClick(e) {
    var linkedVizML = this.linkedVizML;
    var linkedData = d3.select(this).datum();
    linkedVizML.addNode(linkedData);
}

/* event handler for viz node */
function dbclickGeneratedNode(e) {
    var linkedVizML = this.linkedVizML;
    var generatedNodeInfo = d3.select(this).datum();
    linkedVizML.removeNode(generatedNodeInfo["id"]);
}

function dragGenerateNodeStart(e) {
    /* remove the generated link */
    // translate(x, y)
    let translateInitial = d3.select(this).attr("transform").split("(")[1].split(")")[0].split(",").map(x=>parseInt(x, 10));
    this.initialPosition = {x: translateInitial[0], y: translateInitial[1]};
    this.startPosition = {x: d3.event.x, y: d3.event.y};
    this.checkDragged = false;
}

function dragGenerateNode(e) {
    /* update the position of node */
    let currentPosition = {x: d3.event.x, y: d3.event.y};
    if (!this.checkDragged) {
        let moveDistance = Math.pow(currentPosition["x"] - this.startPosition["x"], 2) +
            Math.pow(currentPosition["y"] - this.startPosition["y"], 2);
        if (moveDistance > 10)
            this.checkDragged = true;
    }

    if (this.checkDragged) {
        let presentPosition = {
            x: currentPosition.x - this.startPosition.x + this.initialPosition.x,
            y: currentPosition.y - this.startPosition.y + this.initialPosition.y
        };
        d3.select(this).attr("transform", "translate(" + presentPosition.x + "," + presentPosition.y + ")");
    }
}

function dragGenerateNodeEnd(e) {
    if (this.checkDragged) {
        let linkedData = d3.select(this).datum();
        linkedData["position"]["x"] = d3.event.x - this.startPosition.x + this.initialPosition.x;
        linkedData["position"]["y"] = d3.event.y - this.startPosition.y + this.initialPosition.y;
        d3.select(this)
            .attr("transform", "translate(" + linkedData["position"]["x"] + "," + linkedData["position"]["y"] + ")");
        let linkedVizML = this.linkedVizML;

        /* recreate link */
        linkedData["links"].forEach(function (linkID) {
            linkedVizML.addLink(linkedVizML.linkRecorder.get(linkID).datum());
        });
    }
    this.checkDragged = false;
    this.startPosition = undefined;
    this.initialPosition = undefined;
}

function clickPort(e) {
    var linkedVizML = this.linkedVizML;
    /* add link */
    if (linkedVizML.currentSelection["source"]["nodeID"] === null) {
        /* check link condition, not input port */
        let sourceLinkedData = linkedVizML.nodeRecorder.get(this.linkedNodeId).datum();
        let sourcePort = d3.select(this).datum();
        if (sourcePort !== 1 && sourcePort !== 2) {
            /* add to source */
            /* invariant 2 for transform or layer */
            if (sourceLinkedData["type"] === "transform" || sourceLinkedData["type"] === "layer") {
                let counter = 0;
                let whetherSave = false;
                sourceLinkedData["links"].forEach(function (linkID) {
                    let linkInfo = linkedVizML.linkRecorder.get(linkID).datum();
                    if (linkInfo["source"]["port"] === sourcePort) {
                        counter++;
                        let linkNodeInfo = linkedVizML.nodeRecorder.get(linkInfo["target"]["nodeID"]).datum();
                        if (linkNodeInfo["node"] === "Saver" && linkNodeInfo["type"] === "monitor")
                            whetherSave = true;
                    }
                });
                if (counter > 1 && whetherSave) {
                    alert("Ambiguous when saving a tensor with multiple outbound link copy");
                    return;
                }
            }
            d3.select(this).attr("stroke", "black").attr("stroke-width", 4);
            linkedVizML.currentSelection["source"]["nodeID"] = this.linkedNodeId;
            linkedVizML.currentSelection["source"]["port"] = sourcePort;
        } else {
            alert("Illegal link start from input port !");
        }
    } else {
        /* check link condition, should be input port */
        var generatedLinkInfo = {};
        let sourceNode = linkedVizML.currentSelection["source"]["nodeID"];
        let sourcePort = linkedVizML.currentSelection["source"]["port"];
        let targetNode = this.linkedNodeId;
        let targetPort = d3.select(this).datum();

        generatedLinkInfo["source"] = {nodeID: sourceNode, port: sourcePort};
        generatedLinkInfo["target"] = {nodeID: targetNode, port: targetPort};

        let sourceLinkedData = linkedVizML.nodeRecorder.get(sourceNode).datum();
        let targetLinkedData = linkedVizML.nodeRecorder.get(targetNode).datum();

        if (sourceNode === targetNode) {
            if (sourcePort === targetPort) {
                linkedVizML.currentSelection = {
                    source: {nodeID:null, port:null},
                    target: {nodeID:null, port:null}
                };
                sourceLinkedData["portMap"].get(sourcePort).attr("stroke", "none").attr("stroke-width", 0);
                return;
            } else {
                alert("Node link to itself !")
                return;
            }
        }

        let whetherGenerateLink = false;
        if (targetPort === 1 || targetPort === 2) {
            /* generate link with different color - (to optimizer - loss, track), (to saver), (to other node) */

            if (targetLinkedData["type"] === "optimizer") {
                /* invariant 1: check main input port for leaf invariants, should from input, constant, or detach node, meta port of layer */
                if (targetPort === 1) {
                    /* require gradient */
                    if (sourceLinkedData["type"] === "input" || sourceLinkedData["type"] === "constant") {
                        if (sourcePort === 3) {
                            /* generate link */
                            whetherGenerateLink = true;
                        } else {
                            alert("Use meta port for optimizing input or constant tensor");
                            return;
                        }
                    } else if (sourceLinkedData["type"] === "layer") {
                        if (sourcePort === 3) {
                            /* generate link */
                            whetherGenerateLink = true;
                        } else {
                            alert("Use meta port for optimizing parameters in layer");
                            return;
                        }
                    } else if (sourceLinkedData["node"] === "Detach" && sourceLinkedData["type"] === "transform"){
                        /* generate link */
                        whetherGenerateLink = true;
                    } else {
                        alert("Non-leaf node link to optimizer, can't perform gradient updates");
                        return;
                    }
                    generatedLinkInfo["color"] = "#4daf4a"; // gradient
                } else if (targetPort === 2) {
                    if (sourcePort === 3) {
                        whetherGenerateLink = true;
                    } else {
                        alert("Loss should be the output tensor, not label or others");
                        return;
                    }
                    generatedLinkInfo["color"] = "#e41a1c"; // loss
                }
            } else if (targetLinkedData["node"] === "Saver" && targetLinkedData["type"] === "Manager") {
                /*
                    invariant 2: check end node for safe node, should be from input (not support for dataset, only meta),
                    constant (only meta), output link of (transform or layer, no other out links)
                */
                if (sourceLinkedData["type"] === "input") {
                    if (sourcePort === 3) {
                        /* generate link */
                        if (sourceLinkedData["node"] === "MNIST") {
                            alert("Not support for saving dataset (MNIST)");
                            return;
                        } else {
                            whetherGenerateLink = true;
                        }
                    } else {
                        alert("Use meta port for saving input");
                        return;
                    }
                } else if (sourceLinkedData["type"] === "constant") {
                    if (sourcePort === 3) {
                        /* generate link */
                        whetherGenerateLink = true;
                    } else {
                        alert("Use meta port for saving constant");
                        return;
                    }
                } else if (sourceLinkedData["type"] === "transform" || sourceLinkedData["type"] === "layer") {
                    let counter = 0;
                    sourceLinkedData["links"].forEach(function (linkID) {
                        let linkInfo = linkedVizML.linkRecorder.get(linkID).datum();
                        if (linkInfo["source"]["port"] === sourcePort)
                            counter++;
                    });
                    if (counter > 1) {
                        alert("Ambiguous when saving a tensor with multiple outbound link copy (or duplicate save)");
                        return;
                    } else {
                        whetherGenerateLink = true;
                    }
                }
                generatedLinkInfo["color"] = "#b3cde3";  // save tensor
            } else {
                whetherGenerateLink = true;
                generatedLinkInfo["color"] = "#fb9a99" // forward
            }
        } else {
            alert("Illegal link end at input port!");
            return;
        }

        if (whetherGenerateLink) {
            linkedVizML.currentSelection = {
                source: {nodeID:null, port:null},
                target: {nodeID:null, port:null}
            };
            linkedVizML.addLink(generatedLinkInfo);
        }
    }
    console.log("test for click - port");
}

function clickPanel(e) {
    console.log("test for click - panel");
    var linkedVizML = this.linkedVizML;
    /* check whether current is already selected */
    if (linkedVizML.nodeRecorder.has(this.linkedNodeId)) {
        if (linkedVizML.currentLayerSelection.has(this.linkedNodeId)) {
            d3.select(this).select(".vizNodeRect").style("stroke", "none").style("stroke-width", 0);
            linkedVizML.currentLayerSelection.delete(this.linkedNodeId);
        } else {
            d3.select(this).select(".vizNodeRect").style("stroke", "black").style("stroke-width", 5);
            linkedVizML.currentLayerSelection.add(this.linkedNodeId);
        }
    }
    console.log(linkedVizML.currentLayerSelection);
}

/* event handler for viz link */
function dbclickGeneratedLink() {
    var linkedVizML = this.linkedVizML;
    var generatedLinkInfo = d3.select(this).datum();
    /* remove the info in node */
    linkedVizML.removeLink(generatedLinkInfo["id"]);
}

/* event handler for viz para */
function clickParam(e) {
    var currentText = d3.select(this).text();
    var linkedVizML = this.linkedVizML;
    var linkedNodeId = this.linkedNodeId;
    if (currentText === "▼") {
        /* check current vizParamEnter */
        if (parseInt(linkedVizML.VizParamEnter.style("opacity"), 10) > 0) {
            alert("Please enter the parameter for current node!");
            return
        }
        /* make sure all parameter is entered - paraName, paraValue, paraType, Required */
        linkedVizML.VizParamEnter.transition()
            .style("opacity", .9);
        let generatedNodeInfo = linkedVizML.nodeRecorder.get(linkedNodeId).datum();
        let generatedNodeParam = generatedNodeInfo["parameters"];
        let generatedHTMLText = '<form>';
        generatedHTMLText += generatedHTMLText += ('<label id="source"><b>source</b>:' + generatedNodeInfo["source"] + '</label><br>');
        generatedNodeParam.forEach(function (d) {
            generatedHTMLText += ('<label>' + d["ParaName"]);
            if (d["Required"] === 1)
                generatedHTMLText += "(*)"
            generatedHTMLText += (" - <b>" + d["ParaClass"] + "</b></label>");
            generatedHTMLText += ('<input type="text" id="' + d["ParaName"] + '" value="' + d["ParaValue"] + '"><br>')
        });
        generatedHTMLText += '</form>';

        linkedVizML.VizParamEnter.html(generatedHTMLText)
            .style("left", (d3.event.pageX - 300) + "px")
            .style("top", (d3.event.pageY + 25) + "px")
            .style("width", "250px")
            .style("padding", "5px 5px 5px 8px");
        d3.select(this).text("▲");

    } else if (currentText === "▲") {
        /* make sure all parameter is entered */
        linkedVizML.VizParamEnter.html("")
            .style("width", 0)
            .style("padding", 0)
            .style("opacity", 0);
        d3.select(this).text("▼");
    }
}


/* event handler for viz block */
function dbclickGeneratedBlock() {

}

export { VizML };