import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

import { encodePosCp4m } from "./cp4m-encode.js";

const [inputPath, outputPath, targetText, binWidthText, exponentText] = (
  process.argv.slice(2)
);
if (!inputPath) {
  console.error(
    "usage: node encode.mjs INPUT.pos [OUTPUT.cp4m] [TARGET] [BIN_WIDTH] [EXPONENT]",
  );
  process.exit(2);
}

const input = await readFile(inputPath);
const options = {};
if (targetText) options.targetPoints = Number(targetText);
if (binWidthText) options.binWidthDa = Number(binWidthText);
if (exponentText) options.allocationExponent = Number(exponentText);
const payload = encodePosCp4m(input, options);
if (outputPath && outputPath !== "-") await writeFile(outputPath, payload);
console.log(JSON.stringify({
  inputBytes: input.byteLength,
  outputBytes: payload.byteLength,
  sha256: createHash("sha256").update(payload).digest("hex"),
}));
