import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [workbookPath, previewPath] = process.argv.slice(2);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const check = await workbook.inspect({
  kind: "table",
  range: "Master_Features!A1:AH6",
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 34,
});
console.log(check.ndjson);
const preview = await workbook.render({
  sheetName: "Master_Features",
  range: "A1:AH12",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
