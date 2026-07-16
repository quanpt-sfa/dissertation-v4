import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [inputPath, rowsPath, outputPath] = process.argv.slice(2);
const rows = JSON.parse(await fs.readFile(rowsPath, "utf8"));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const sheet = workbook.worksheets.getItem("Master_Features");
const headers = sheet.getRange("A1:AH1").values[0];
const byId = new Map(rows.map((row) => [row.feature_id, row]));
for (let rowIndex = 1; rowIndex <= rows.length; rowIndex += 1) {
  const featureId = sheet.getCell(rowIndex, 1).values[0][0];
  const audit = byId.get(featureId);
  if (!audit) throw new Error(`missing audit row for ${featureId}`);
  for (const header of headers.slice(18)) {
    const column = headers.indexOf(header);
    sheet.getCell(rowIndex, column).values = [[audit[header] ?? null]];
  }
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
