# KaTeX 0.16.10, vendored

The math in the Forge Assistant's panel is typeset by this copy of KaTeX
(MIT licence, `LICENSE` beside this file). It is the same version Gradio 4.40
bundles for LLM Studio's transcript, so a formula looks the same in the panel
and in the tab.

What is here, and nothing else, is taken unchanged from the npm package
`katex@0.16.10` (`https://registry.npmjs.org/katex/-/katex-0.16.10.tgz`,
integrity `sha512-ZiqaC04tp2O5utMsl2TEZTXxa6WSC4yo0fv5ML++D3QZv/vx2Mct0mTlRx3O+uUkjfuAgOkzsCmq5MiUEsDDdA==`):

- `katex.min.js` and `katex.min.css`, from `dist/`;
- the twenty `woff2` fonts, from `dist/fonts/` -- the stylesheet also names a
  `woff` and a `ttf` of each, which every browser this extension supports never
  asks for once the `woff2` loads;
- `LICENSE`.

`manifest.json` lists each file with its SHA-256. `mc_katex.py` serves exactly
those files, at `/model-chain/katex/0.16.10/<file>`, and nothing else from this
folder; `tests/test_katex_route.py` checks every digest, so an edit here fails
a test. Nothing loads them until the panel draws a message with math in it
(`javascript/forge_assistant_markdown.js`).

To move to another version: take the same files from that version's package,
write the manifest again with their digests, and change the version in
`mc_katex.VERSION` and in the panel's script, which a test holds equal.
Gradio's own copy is the one LLM Studio uses; keeping the two the same keeps
the two surfaces alike, and their stylesheets, which both load, agree.
