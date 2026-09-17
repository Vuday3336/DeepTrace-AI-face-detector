import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../auth/AuthContext";
import { makeFace, makePrediction } from "../test/fixtures";
import { AnalyzePage } from "./AnalyzePage";

const renderPage = () =>
  render(
    <MemoryRouter>
      <AuthProvider>
        <AnalyzePage />
      </AuthProvider>
    </MemoryRouter>,
  );

const upload = (file: File) => fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });

describe("AnalyzePage", () => {
  it("uploads, shows one card per face, and never stores by default", async () => {
    const prediction = makePrediction([makeFace(), makeFace({ face_index: 1, label: "REAL", prob_ai_generated: 0.1 })]);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(prediction), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    renderPage();
    upload(new File(["img"], "photo.jpg", { type: "image/jpeg" }));

    expect(await screen.findByText("Likely AI-generated")).toBeInTheDocument();
    expect(screen.getByText("Likely real")).toBeInTheDocument();
    expect(screen.getByText(/2 faces/)).toBeInTheDocument();
    expect(screen.getByText(/not stored/)).toBeInTheDocument();
    expect(String(fetchSpy.mock.calls[0]![0])).toContain("store=false");
  });

  it("shows a friendly message when no face is found", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: { code: "NO_FACE_DETECTED", message: "x", request_id: "r" } }), { status: 422 }),
    );
    renderPage();
    upload(new File(["img"], "landscape.png", { type: "image/png" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/No face found/);
  });

  it("rejects unsupported files before calling the API", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    renderPage();
    upload(new File(["gif"], "anim.gif", { type: "image/gif" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/JPG, PNG or WebP/));
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
