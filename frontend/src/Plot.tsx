// Basic bundle (scatter/bar/pie, ~1 MB) instead of full Plotly (~4.9 MB): the backend only emits px.bar / px.line.
// Add a trace type here only if a chart ever needs one outside that set.
import createPlotlyComponent from "react-plotly.js/factory";
// @ts-expect-error — the basic bundle ships no type declarations
import Plotly from "plotly.js-basic-dist-min";

export default createPlotlyComponent(Plotly);
