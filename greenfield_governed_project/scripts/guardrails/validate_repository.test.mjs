/*
PROV: GREENFIELD.TEST.GUARDRAILS.01
REQ: GREENFIELD-TEST-001
WHY: Unit tests for guardrails validation logic to ensure governance rules are enforced correctly.
*/

import { describe, it } from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");

describe("Guardrails validation", () => {
  describe("Markdown generation validation", () => {
    it("should pass when all markdown is generated", () => {
      // Run guardrails on the actual repo
      const result = execSync("npm run guardrails", { 
        cwd: repoRoot,
        encoding: "utf8",
        stdio: "pipe"
      });
      assert.ok(result.includes("✓ guardrails ok"));
    });

    it("should detect hand-edited markdown files", () => {
      const testMdPath = path.join(repoRoot, "test_handwritten.md");
      try {
        // Create a markdown file without generated frontmatter
        fs.writeFileSync(testMdPath, "# Test\n\nThis is hand-written.");
        
        // Run guardrails and expect failure
        assert.throws(() => {
          execSync("npm run guardrails", { 
            cwd: repoRoot,
            encoding: "utf8",
            stdio: "pipe"
          });
        }, /markdown_generation_violation/);
      } finally {
        // Cleanup
        if (fs.existsSync(testMdPath)) fs.unlinkSync(testMdPath);
      }
    });
  });

  describe("Requirements validation", () => {
    it("should validate requirement IDs are unique", () => {
      const projectPath = path.join(repoRoot, "spec", "project.json");
      const project = JSON.parse(fs.readFileSync(projectPath, "utf8"));
      const reqsPath = path.join(repoRoot, project.requirements_source);
      const reqs = JSON.parse(fs.readFileSync(reqsPath, "utf8"));
      
      if (reqs.type === "requirements_index") {
        const allIds = new Set();
        for (const file of reqs.files) {
          const area = JSON.parse(fs.readFileSync(path.join(repoRoot, file), "utf8"));
          for (const r of area.requirements) {
            assert.ok(!allIds.has(r.id), `Duplicate requirement ID: ${r.id}`);
            allIds.add(r.id);
          }
        }
      }
    });

    it("should validate requirements have required fields", () => {
      const projectPath = path.join(repoRoot, "spec", "project.json");
      const project = JSON.parse(fs.readFileSync(projectPath, "utf8"));
      const reqsPath = path.join(repoRoot, project.requirements_source);
      const reqs = JSON.parse(fs.readFileSync(reqsPath, "utf8"));
      
      if (reqs.type === "requirements_index") {
        for (const file of reqs.files) {
          const area = JSON.parse(fs.readFileSync(path.join(repoRoot, file), "utf8"));
          for (const r of area.requirements) {
            assert.ok(r.id, "Requirement missing ID");
            assert.ok(r.title, `Requirement ${r.id} missing title`);
            assert.ok(r.status, `Requirement ${r.id} missing status`);
            assert.ok(Array.isArray(r.guardrails) && r.guardrails.length > 0, 
              `Requirement ${r.id} missing guardrails`);
          }
        }
      }
    });

    it("should validate requirement tracking status", () => {
      const projectPath = path.join(repoRoot, "spec", "project.json");
      const project = JSON.parse(fs.readFileSync(projectPath, "utf8"));
      const reqsPath = path.join(repoRoot, project.requirements_source);
      const reqs = JSON.parse(fs.readFileSync(reqsPath, "utf8"));
      
      if (reqs.type === "requirements_index") {
        for (const file of reqs.files) {
          const area = JSON.parse(fs.readFileSync(path.join(repoRoot, file), "utf8"));
          for (const r of area.requirements) {
            const impl = r.tracking?.implementation;
            if (impl) {
              assert.ok(["todo", "done"].includes(impl), 
                `Requirement ${r.id} has invalid tracking.implementation: ${impl}`);
            }
          }
        }
      }
    });
  });

  describe("Intent validation", () => {
    it("should validate intent files match their IDs", () => {
      const intentsDir = path.join(repoRoot, "spec", "intents");
      const files = fs.readdirSync(intentsDir).filter(f => f.endsWith(".json"));
      
      for (const file of files) {
        const intent = JSON.parse(fs.readFileSync(path.join(intentsDir, file), "utf8"));
        const expectedFilename = `${intent.intent_id}.json`;
        assert.strictEqual(file, expectedFilename, 
          `Intent filename mismatch: ${file} should be ${expectedFilename}`);
      }
    });

    it("should validate intents have required fields", () => {
      const intentsDir = path.join(repoRoot, "spec", "intents");
      const files = fs.readdirSync(intentsDir).filter(f => f.endsWith(".json"));
      
      for (const file of files) {
        const intent = JSON.parse(fs.readFileSync(path.join(intentsDir, file), "utf8"));
        assert.ok(intent.intent_id, "Intent missing intent_id");
        assert.ok(intent.title, `Intent ${intent.intent_id} missing title`);
        assert.ok(intent.status, `Intent ${intent.intent_id} missing status`);
        assert.ok(Array.isArray(intent.requirements_in_scope), 
          `Intent ${intent.intent_id} requirements_in_scope must be array`);
        assert.ok(Array.isArray(intent.task_ids_planned), 
          `Intent ${intent.intent_id} task_ids_planned must be array`);
      }
    });

    it("should validate intent runbooks structure", () => {
      const intentsDir = path.join(repoRoot, "spec", "intents");
      const files = fs.readdirSync(intentsDir).filter(f => f.endsWith(".json"));
      
      for (const file of files) {
        const intent = JSON.parse(fs.readFileSync(path.join(intentsDir, file), "utf8"));
        if (intent.runbooks && intent.runbooks !== null) {
          assert.ok(intent.runbooks.decision, 
            `Intent ${intent.intent_id} runbooks missing decision`);
          assert.ok(["none", "inline", "external", "update"].includes(intent.runbooks.decision),
            `Intent ${intent.intent_id} runbooks has invalid decision: ${intent.runbooks.decision}`);
          assert.ok(intent.runbooks.notes !== undefined,
            `Intent ${intent.intent_id} runbooks missing notes`);
          assert.ok(Array.isArray(intent.runbooks.paths_mdt),
            `Intent ${intent.intent_id} runbooks.paths_mdt must be array`);
        }
      }
    });
  });

  describe("Task validation", () => {
    it("should validate task files exist for planned tasks", () => {
      const intentsDir = path.join(repoRoot, "spec", "intents");
      const tasksDir = path.join(repoRoot, "spec", "tasks");
      
      if (!fs.existsSync(tasksDir)) return; // Skip if no tasks yet
      
      const intentFiles = fs.readdirSync(intentsDir).filter(f => f.endsWith(".json"));
      
      for (const file of intentFiles) {
        const intent = JSON.parse(fs.readFileSync(path.join(intentsDir, file), "utf8"));
        for (const taskId of intent.task_ids_planned || []) {
          const taskPath = path.join(tasksDir, `${taskId}.json`);
          assert.ok(fs.existsSync(taskPath), 
            `Task ${taskId} planned in ${intent.intent_id} but file not found`);
        }
      }
    });

    it("should validate tasks have required fields", () => {
      const tasksDir = path.join(repoRoot, "spec", "tasks");
      if (!fs.existsSync(tasksDir)) return;
      
      const files = fs.readdirSync(tasksDir).filter(f => f.endsWith(".json"));
      
      for (const file of files) {
        const task = JSON.parse(fs.readFileSync(path.join(tasksDir, file), "utf8"));
        assert.ok(task.task_id, "Task missing task_id");
        assert.ok(task.intent_id, `Task ${task.task_id} missing intent_id`);
        assert.ok(task.title, `Task ${task.task_id} missing title`);
        assert.ok(task.status, `Task ${task.task_id} missing status`);
      }
    });
  });

  describe("Generated files validation", () => {
    it("should validate scope.json is generated correctly", () => {
      const intentsDir = path.join(repoRoot, "spec", "intents");
      const files = fs.readdirSync(intentsDir).filter(f => f.endsWith(".json"));
      
      for (const file of files) {
        const intent = JSON.parse(fs.readFileSync(path.join(intentsDir, file), "utf8"));
        const scopePath = path.join(repoRoot, "status", "intents", intent.intent_id, "scope.json");
        
        if (fs.existsSync(scopePath)) {
          const scope = JSON.parse(fs.readFileSync(scopePath, "utf8"));
          assert.strictEqual(scope.intent_id, intent.intent_id);
          assert.ok(Array.isArray(scope.requirements_in_scope));
          assert.ok(Array.isArray(scope.task_ids_planned));
          assert.ok(scope.close_gate);
          assert.ok(Array.isArray(scope.close_gate.commands));
        }
      }
    });

    it("should validate internal_intents.json structure", () => {
      const feedPath = path.join(repoRoot, "status", "portal", "internal_intents.json");
      assert.ok(fs.existsSync(feedPath), "internal_intents.json should exist");
      
      const feed = JSON.parse(fs.readFileSync(feedPath, "utf8"));
      assert.strictEqual(feed.generated, true);
      assert.ok(feed.source);
      assert.ok(Array.isArray(feed.intents));
      
      for (const intent of feed.intents) {
        assert.ok(intent.intent_id);
        assert.ok(intent.title);
        assert.ok(intent.status);
        assert.ok(Array.isArray(intent.requirements_in_scope));
        assert.ok(Array.isArray(intent.tasks));
      }
    });
  });

  describe("REQ tag enforcement", () => {
    it("should find REQ tags in code files", () => {
      const scriptsDir = path.join(repoRoot, "scripts");
      let foundReqTag = false;
      
      function scanDir(dir) {
        for (const item of fs.readdirSync(dir, { withFileTypes: true })) {
          const fullPath = path.join(dir, item.name);
          if (item.isDirectory() && item.name !== "node_modules") {
            scanDir(fullPath);
          } else if (item.isFile() && (item.name.endsWith(".mjs") || item.name.endsWith(".js"))) {
            const content = fs.readFileSync(fullPath, "utf8");
            if (/REQ:\s*[A-Z0-9_-]+-[0-9]+/.test(content)) {
              foundReqTag = true;
            }
          }
        }
      }
      
      scanDir(scriptsDir);
      assert.ok(foundReqTag, "Should find at least one REQ tag in codebase");
    });
  });

  describe("Schema validation", () => {
    it("should validate JSON schemas exist", () => {
      const schemasDir = path.join(repoRoot, "spec", "schemas");
      assert.ok(fs.existsSync(schemasDir), "schemas directory should exist");
      
      const expectedSchemas = [
        "intent_scope.schema.json",
        "work_packages.schema.json",
        "internal_intents.schema.json"
      ];
      
      for (const schema of expectedSchemas) {
        const schemaPath = path.join(schemasDir, schema);
        assert.ok(fs.existsSync(schemaPath), `Schema ${schema} should exist`);
        
        // Validate it's valid JSON
        const content = fs.readFileSync(schemaPath, "utf8");
        assert.doesNotThrow(() => JSON.parse(content), 
          `Schema ${schema} should be valid JSON`);
      }
    });
  });

  describe("Evidence format validation", () => {
    it("should validate run.json structure in audit evidence", () => {
      const auditDir = path.join(repoRoot, "status", "audit");
      if (!fs.existsSync(auditDir)) return;
      
      const intents = fs.readdirSync(auditDir).filter(d => 
        fs.statSync(path.join(auditDir, d)).isDirectory()
      );
      
      for (const intentId of intents) {
        const runsDir = path.join(auditDir, intentId, "runs");
        if (!fs.existsSync(runsDir)) continue;
        
        const runs = fs.readdirSync(runsDir).filter(d =>
          fs.statSync(path.join(runsDir, d)).isDirectory()
        );
        
        for (const runId of runs) {
          const runJsons = [];
          
          function findRunJson(dir) {
            for (const item of fs.readdirSync(dir, { withFileTypes: true })) {
              const fullPath = path.join(dir, item.name);
              if (item.isDirectory()) {
                findRunJson(fullPath);
              } else if (item.isFile() && item.name === "run.json") {
                runJsons.push(fullPath);
              }
            }
          }
          
          findRunJson(path.join(runsDir, runId));
          
          for (const runJsonPath of runJsons) {
            const runData = JSON.parse(fs.readFileSync(runJsonPath, "utf8"));
            assert.ok(runData.intent_id, `run.json missing intent_id: ${runJsonPath}`);
            assert.ok(runData.run_id, `run.json missing run_id: ${runJsonPath}`);
            // stage is optional for backwards compatibility
            assert.ok(runData.command, `run.json missing command: ${runJsonPath}`);
            assert.ok(runData.timestamp_end, `run.json missing timestamp_end: ${runJsonPath}`);
            assert.ok(typeof runData.exit_code === "number", `run.json exit_code must be number: ${runJsonPath}`);
          }
        }
      }
    });
  });
});
