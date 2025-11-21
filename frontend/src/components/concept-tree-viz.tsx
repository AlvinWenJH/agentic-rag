import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
type ConceptNode = { name: string; children?: ConceptNode[] }

interface Props {
    data: ConceptNode | null;
}

const ConceptTreeViz: React.FC<Props> = ({ data }) => {
    const svgRef = useRef<SVGSVGElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!data || !svgRef.current || !wrapperRef.current) return;

        const width = wrapperRef.current.clientWidth;
        const height = wrapperRef.current.clientHeight;

        // Clear previous
        d3.select(svgRef.current).selectAll("*").remove();

        const svg = d3.select(svgRef.current)
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", `translate(${width / 2}, 50)`); // Top-down

        // Convert data to hierarchy
        const root = d3.hierarchy(data);
        const treeLayout = d3.tree<ConceptNode>().size([width - 100, height - 150]);

        treeLayout(root);

        // Links
        svg.selectAll(".link")
            .data(root.links())
            .enter()
            .append("path")
            .attr("class", "link")
            .attr("fill", "none")
            .attr("stroke", "#e879f9") // fuchsia-400
            .attr("stroke-opacity", 0.4)
            .attr("stroke-width", 1.5)
            .attr("d", d3.linkVertical()
                .x((d: any) => d.x)
                .y((d: any) => d.y) as any
            );

        // Nodes
        const node = svg.selectAll(".node")
            .data(root.descendants())
            .enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", (d: any) => `translate(${d.x},${d.y})`);

        node.append("circle")
            .attr("r", 6)
            .attr("fill", "#c026d3") // fuchsia-600
            .attr("stroke", "#fff")
            .attr("stroke-width", 2);

        node.append("text")
            .attr("dy", (d: any) => d.children ? -15 : 20)
            .style("text-anchor", "middle")
            .text((d: any) => d.data.name)
            .style("fill", "#e2e8f0")
            .style("font-size", "12px")
            .style("font-family", "Inter, sans-serif")
            .style("text-shadow", "0px 0px 4px #000");

        // Tooltip behavior could go here, but keeping it visual for now.

    }, [data]);

    if (!data) return null;

    return (
        <div ref={wrapperRef} className="w-full h-full min-h-[600px] bg-zinc-900/50 backdrop-blur-sm rounded-xl border border-zinc-800 overflow-hidden shadow-2xl">
            <svg ref={svgRef} className="w-full h-full"></svg>
        </div>
    );
};

export default ConceptTreeViz;