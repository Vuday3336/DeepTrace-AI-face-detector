import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { makeFace, makePrediction } from "../test/fixtures";
import { HeatmapViewer } from "./HeatmapViewer";
import { ProbabilityMeter } from "./ProbabilityMeter";
import { ResultCard } from "./ResultCard";

describe("ProbabilityMeter", () => {
  it("shows the probability and positions the marker", () => {
    render(<ProbabilityMeter prob={0.87} threshold={0.62} delta={0.05} />);
    expect(screen.getByTestId("prob-value")).toHaveTextContent("87%");
    expect(screen.getByTestId("prob-marker")).toHaveStyle({ left: "87%" });
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "87");
    expect(screen.getByText("threshold 62%")).toBeInTheDocument();
  });

  it("omits the uncertain band when delta is 0", () => {
    render(<ProbabilityMeter prob={0.4} threshold={0.5} delta={0} />);
    expect(screen.queryByTestId("uncertain-band")).not.toBeInTheDocument();
  });
});

describe("HeatmapViewer", () => {
  it("changes overlay opacity with the slider", () => {
    const face = makeFace();
    render(<HeatmapViewer cropBase64={face.face_crop_png_base64} heatmapBase64={face.heatmap_png_base64} method="gradcam" />);
    const overlay = screen.getByTestId("heatmap-overlay");
    expect(overlay).toHaveStyle({ opacity: "0.6" });
    fireEvent.change(screen.getByLabelText(/heatmap opacity/i), { target: { value: "0.2" } });
    expect(overlay).toHaveStyle({ opacity: "0.2" });
    expect(screen.getByText(/Grad-CAM/)).toBeInTheDocument();
  });

  it("explains that attention rollout is class-agnostic", () => {
    render(<HeatmapViewer cropBase64="x" heatmapBase64="y" method="attention_rollout" />);
    expect(screen.getByText(/class-agnostic/)).toBeInTheDocument();
  });

  it("handles a missing heatmap", () => {
    render(<HeatmapViewer cropBase64="x" heatmapBase64={null} method="gradcam" />);
    expect(screen.queryByTestId("heatmap-overlay")).not.toBeInTheDocument();
    expect(screen.getByText("Heatmap not requested")).toBeInTheDocument();
  });
});

describe("ResultCard", () => {
  it.each([
    ["AI_GENERATED", "Likely AI-generated"],
    ["REAL", "Likely real"],
    ["UNCERTAIN", "Uncertain"],
  ] as const)("renders the %s verdict with a hedged explanation", (label, text) => {
    const prediction = makePrediction([makeFace({ label })]);
    render(<ResultCard face={prediction.faces[0]!} model={prediction.model} total={1} />);
    expect(screen.getByText(text)).toBeInTheDocument();
    expect(screen.getByText(/not (prove|proof)|inconclusive/i)).toBeInTheDocument();
  });
});
