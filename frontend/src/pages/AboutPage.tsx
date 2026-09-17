import { Link } from "react-router-dom";
import { Disclaimer, Panel } from "../components/ui";

const STEPS = [
  ["Face detection", "MTCNN finds every face and crops a square around it with a margin of context."],
  ["Canonical preprocessing", "Each crop is resized to 224×224 and re-encoded as JPEG, exactly like the training data, and all metadata is discarded."],
  ["Classification", "A neural network trained on real (FFHQ) and AI-generated (StyleGAN) faces outputs a score, calibrated so probabilities are meaningful."],
  ["Decision", "The score is compared to a threshold chosen to limit false accusations; scores near the threshold are reported as uncertain."],
  ["Explanation", "A heatmap highlights the image regions that most influenced the score."],
];

export function AboutPage() {
  return (
    <div className="max-w-3xl space-y-6">
      <h1 className="text-2xl font-semibold text-white">How DeepTrace works</h1>
      <Disclaimer />
      <Panel>
        <ol className="space-y-4">
          {STEPS.map(([title, text], i) => (
            <li key={title} className="flex gap-4">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent/15 font-mono text-sm text-accent">{i + 1}</span>
              <div>
                <h2 className="font-medium text-white">{title}</h2>
                <p className="text-sm text-slate-400">{text}</p>
              </div>
            </li>
          ))}
        </ol>
      </Panel>
      <Panel className="space-y-3 text-sm text-slate-300">
        <h2 className="font-semibold text-white">Where it fails</h2>
        <p>
          Detectors learn the fingerprints of the generators they were trained on. Faces from newer diffusion models,
          heavily compressed images, screenshots and edited photos are all harder. The{" "}
          <Link to="/model" className="text-accent underline">model page</Link> reports measured accuracy on those cases
          instead of a single headline number.
        </p>
      </Panel>
      <Panel className="space-y-3 text-sm text-slate-300">
        <h2 className="font-semibold text-white">Privacy</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>By default, uploads are processed in memory and discarded after the response.</li>
          <li>Saving to history is opt-in and requires an account; stored images have metadata (e.g. GPS) removed and expire automatically.</li>
          <li>You can delete any saved analysis at any time.</li>
        </ul>
      </Panel>
      <Panel className="space-y-3 text-sm text-slate-300">
        <h2 className="font-semibold text-white">Responsible use</h2>
        <p>
          Never use DeepTrace alone to accuse someone of faking a photo or to decide someone's identity is fraudulent.
          A “likely AI-generated” result is a reason to look closer — at the source, context and other evidence — not
          a conclusion. Training data (FFHQ) is licensed for non-commercial use only.
        </p>
      </Panel>
    </div>
  );
}
