import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";
const [csvPath, outPath] = process.argv.slice(2);
const rows = await fs.readFile(csvPath, "utf8");
const wb = await Workbook.fromCSV(rows, { sheetName: "Mapping_Proposal" });
const ws = wb.worksheets.getItem("Mapping_Proposal");
ws.freezePanes.freezeRows(1); ws.showGridLines=false;
ws.getUsedRange().format.wrapText=true; ws.getRange("A1:Z1").format={fill:"#0F766E",font:{bold:true,color:"#FFFFFF"}};
const headers = ws.getUsedRange().getRow(0).values[0];
const decisionCol = headers.indexOf("researcher_decision");
if (decisionCol >= 0) ws.getRangeByIndexes(1, decisionCol, 211, 1).dataValidation = {rule:{type:"list",values:["APPROVE_EXACT_MAPPING","APPROVE_VAS_MAPPING","APPROVE_ADAPTED_ROBUSTNESS_ONLY","APPROVE_LABEL_MECHANISM_ONLY","REQUEST_ADDITIONAL_EVIDENCE","REQUEST_DATA_COLLECTION","BLOCK","DEFER"]}};
for(const name of ["Overview","Source_Inventory","Competing_Mappings","Coverage","Temporal_Leakage","Confirmed_Gaps","Researcher_Decisions"]){const s=wb.worksheets.add(name);s.getRange("A1").values=[[name]];s.getRange("A1").format={fill:"#0F766E",font:{bold:true,color:"#FFFFFF"}};}
const out=await SpreadsheetFile.exportXlsx(wb);await out.save(outPath);
