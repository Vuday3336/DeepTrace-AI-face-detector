import { describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./client";

const jsonResponse = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("api client", () => {
  it("returns parsed JSON on success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { status: "ok", model_loaded: true }));
    await expect(api.health()).resolves.toMatchObject({ status: "ok" });
  });

  it("maps the error envelope to ApiError with code and request id", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(422, { error: { code: "NO_FACE_DETECTED", message: "No face", request_id: "abc" } }),
    );
    const err = await api.predict(new File(["x"], "a.jpg"), { store: false, explain: true }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 422, code: "NO_FACE_DETECTED", requestId: "abc" });
  });

  it("reports network failures clearly", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(api.health()).rejects.toMatchObject({ code: "NETWORK_ERROR" });
  });

  it("sends the upload as multipart with privacy flags and bearer token", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { faces: [] }));
    await api.predict(new File(["x"], "a.jpg", { type: "image/jpeg" }), { store: true, explain: false }, "tok");
    const [url, init] = spy.mock.calls[0]!;
    expect(String(url)).toContain("/api/v1/predict?store=true&explain=false");
    expect(init?.body).toBeInstanceOf(FormData);
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer tok");
  });

  it("handles non-JSON error bodies", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("<html>502</html>", { status: 502 }));
    await expect(api.health()).rejects.toMatchObject({ status: 502, code: "HTTP_ERROR" });
  });
});
