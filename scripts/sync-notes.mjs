import { copyFile, mkdir, readdir, rm, stat } from "node:fs/promises"
import path from "node:path"

const sourceRoot = "/Users/chenshi/Desktop/notes"
const contentRoot = path.resolve("content")

const excludedNames = new Set(["CLAUDE.md", "AIMbench", "AIM-bench", "组会", "模板"])
const protectedRootNames = new Set(["index.md", ".gitkeep"])

function shouldSkip(name) {
  return name.startsWith(".") || excludedNames.has(name)
}

async function syncDir(sourceDir, targetDir, isRoot = false) {
  await mkdir(targetDir, { recursive: true })

  const sourceEntries = await readdir(sourceDir, { withFileTypes: true })
  const allowedNames = new Set(sourceEntries.filter((entry) => !shouldSkip(entry.name)).map((entry) => entry.name))
  const targetEntries = await readdir(targetDir, { withFileTypes: true })

  for (const entry of targetEntries) {
    if (isRoot && protectedRootNames.has(entry.name)) {
      continue
    }

    if (!allowedNames.has(entry.name)) {
      await rm(path.join(targetDir, entry.name), { recursive: true, force: true })
    }
  }

  for (const entry of sourceEntries) {
    if (shouldSkip(entry.name)) {
      continue
    }

    const sourcePath = path.join(sourceDir, entry.name)
    const targetPath = path.join(targetDir, entry.name)

    if (entry.isDirectory()) {
      await syncDir(sourcePath, targetPath)
      continue
    }

    const fileStat = await stat(sourcePath)
    if (!fileStat.isFile()) {
      continue
    }

    await mkdir(path.dirname(targetPath), { recursive: true })
    await copyFile(sourcePath, targetPath)
  }
}

await syncDir(sourceRoot, contentRoot, true)
console.log(`Synced notes from ${sourceRoot} to ${contentRoot}`)
