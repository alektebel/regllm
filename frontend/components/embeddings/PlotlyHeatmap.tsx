"use client";

import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

interface Props {
  ids: string[];
  matrix: number[][];
}

export default function PlotlyHeatmap({ ids, matrix }: Props) {
  const labels = ids.map((id) => {
    const parts = id.split("#");
    return parts[0].split("/").pop() || id;
  });

  return (
    <Plot
      data={[
        {
          type: "heatmap",
          z: matrix,
          x: labels,
          y: labels,
          colorscale: "Viridis",
          hovertemplate: "%{x}<br>%{y}<br>cosine: %{z:.3f}<extra></extra>",
        },
      ]}
      layout={{
        autosize: true,
        margin: { l: 100, r: 20, t: 20, b: 100 },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#a1a1aa", size: 9 },
        xaxis: { tickangle: -45, tickfont: { size: 8 } },
        yaxis: { tickfont: { size: 8 } },
      }}
      config={{ displaylogo: false, responsive: true }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
