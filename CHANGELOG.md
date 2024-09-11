# Changelog

## [v0.0.6](https://github.com/tlambert03/microsim/tree/v0.0.6) (2024-09-10)

[Full Changelog](https://github.com/tlambert03/microsim/compare/v0.0.5...v0.0.6)

**Merged pull requests:**

- fix: fix key again [\#77](https://github.com/tlambert03/microsim/pull/77) ([tlambert03](https://github.com/tlambert03))
- fix: typos [\#75](https://github.com/tlambert03/microsim/pull/75) ([tlambert03](https://github.com/tlambert03))
- refactor: rewrite simulation stages, improve optical config spectral considerations [\#74](https://github.com/tlambert03/microsim/pull/74) ([tlambert03](https://github.com/tlambert03))
- ci\(pre-commit.ci\): autoupdate [\#72](https://github.com/tlambert03/microsim/pull/72) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- refactor: remove pint from models [\#71](https://github.com/tlambert03/microsim/pull/71) ([tlambert03](https://github.com/tlambert03))
- Fix cosem images download error [\#69](https://github.com/tlambert03/microsim/pull/69) ([veegalinova](https://github.com/veegalinova))
- fix: allowing None type for 'CosemDataset' attributes [\#67](https://github.com/tlambert03/microsim/pull/67) ([federico-carrara](https://github.com/federico-carrara))
- Implementing Illumination/Irradiance as a \(W, C, Z, Y, X\) array [\#63](https://github.com/tlambert03/microsim/pull/63) ([federico-carrara](https://github.com/federico-carrara))
- ci\(pre-commit.ci\): autoupdate [\#60](https://github.com/tlambert03/microsim/pull/60) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))

## [v0.0.5](https://github.com/tlambert03/microsim/tree/v0.0.5) (2024-07-08)

[Full Changelog](https://github.com/tlambert03/microsim/compare/v0.0.4...v0.0.5)

**Merged pull requests:**

- fix: fix cosem fetch [\#65](https://github.com/tlambert03/microsim/pull/65) ([tlambert03](https://github.com/tlambert03))
- Documentation for excitation and emission code + a bug fix. [\#59](https://github.com/tlambert03/microsim/pull/59) ([ashesh-0](https://github.com/ashesh-0))
- add init\_forbid\_extra in pydantic.mypy settings [\#58](https://github.com/tlambert03/microsim/pull/58) ([tlambert03](https://github.com/tlambert03))
- fix: fixed typo \(from 'placemen' to 'placement'\) [\#57](https://github.com/tlambert03/microsim/pull/57) ([federico-carrara](https://github.com/federico-carrara))

## [v0.0.4](https://github.com/tlambert03/microsim/tree/v0.0.4) (2024-06-20)

[Full Changelog](https://github.com/tlambert03/microsim/compare/v0.0.3...v0.0.4)

**Fixed bugs:**

- fix: fix for redhat [\#56](https://github.com/tlambert03/microsim/pull/56) ([tlambert03](https://github.com/tlambert03))

## [v0.0.3](https://github.com/tlambert03/microsim/tree/v0.0.3) (2024-06-17)

[Full Changelog](https://github.com/tlambert03/microsim/compare/v0.0.2...v0.0.3)

**Implemented enhancements:**

- feat: pull neuron ground truths from allen brain [\#48](https://github.com/tlambert03/microsim/pull/48) ([tlambert03](https://github.com/tlambert03))
- feat: support multi-channel, multi-fluorophore simulations [\#35](https://github.com/tlambert03/microsim/pull/35) ([ashesh-0](https://github.com/ashesh-0))

**Merged pull requests:**

- fix: remove caching on bins [\#53](https://github.com/tlambert03/microsim/pull/53) ([tlambert03](https://github.com/tlambert03))
- build: pin numpy to \<2 until next tensorstore release [\#51](https://github.com/tlambert03/microsim/pull/51) ([tlambert03](https://github.com/tlambert03))
- feat: add cosem models, downloading, caching, binning, and tests [\#50](https://github.com/tlambert03/microsim/pull/50) ([tlambert03](https://github.com/tlambert03))
- feat: add ndview convenience [\#49](https://github.com/tlambert03/microsim/pull/49) ([tlambert03](https://github.com/tlambert03))
- docs: Overview of various stages. [\#46](https://github.com/tlambert03/microsim/pull/46) ([tlambert03](https://github.com/tlambert03))
- ci: try self-hosted for cupy [\#45](https://github.com/tlambert03/microsim/pull/45) ([tlambert03](https://github.com/tlambert03))
- Establish dimension conventions, pass dimensions through more explicitly [\#44](https://github.com/tlambert03/microsim/pull/44) ([tlambert03](https://github.com/tlambert03))
- refactor: remove DataArray, use xarray directly, vendor xarray\_jax from deepmind/graphcast [\#43](https://github.com/tlambert03/microsim/pull/43) ([tlambert03](https://github.com/tlambert03))
- ci\(pre-commit.ci\): autoupdate [\#42](https://github.com/tlambert03/microsim/pull/42) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- feat: add caching [\#41](https://github.com/tlambert03/microsim/pull/41) ([tlambert03](https://github.com/tlambert03))
- feat: allow loading optical configs from fpbase, general refactor of optical configs [\#40](https://github.com/tlambert03/microsim/pull/40) ([tlambert03](https://github.com/tlambert03))
- feat: cosem Sample [\#18](https://github.com/tlambert03/microsim/pull/18) ([tlambert03](https://github.com/tlambert03))

## [v0.0.2](https://github.com/tlambert03/microsim/tree/v0.0.2) (2024-05-28)

[Full Changelog](https://github.com/tlambert03/microsim/compare/v0.0.1...v0.0.2)

**Merged pull requests:**

- fix cupy [\#38](https://github.com/tlambert03/microsim/pull/38) ([tlambert03](https://github.com/tlambert03))
- feat: add pint quantities to more fields [\#37](https://github.com/tlambert03/microsim/pull/37) ([tlambert03](https://github.com/tlambert03))
- feat: establish some general Annotated\[..., Validator\(\)\] patterns.  add `Fluorophore.from_fpbase` [\#36](https://github.com/tlambert03/microsim/pull/36) ([tlambert03](https://github.com/tlambert03))
- feat: add emission event example, and add lifetime to state from fpbase, test examples [\#34](https://github.com/tlambert03/microsim/pull/34) ([tlambert03](https://github.com/tlambert03))
- feat: add direct from ground truth [\#32](https://github.com/tlambert03/microsim/pull/32) ([tlambert03](https://github.com/tlambert03))
- feat: swappable fft backends, including torch and jax [\#30](https://github.com/tlambert03/microsim/pull/30) ([tlambert03](https://github.com/tlambert03))

## [v0.0.1](https://github.com/tlambert03/microsim/tree/v0.0.1) (2024-05-07)

[Full Changelog](https://github.com/tlambert03/microsim/compare/e1efcfbf80fbc72153c4769aa3cd59bf7b654b09...v0.0.1)

**Merged pull requests:**

- feat: higher level psf caching [\#28](https://github.com/tlambert03/microsim/pull/28) ([tlambert03](https://github.com/tlambert03))
- Update ci, prep for release to pypi [\#27](https://github.com/tlambert03/microsim/pull/27) ([tlambert03](https://github.com/tlambert03))
- feat: cache psf [\#26](https://github.com/tlambert03/microsim/pull/26) ([tlambert03](https://github.com/tlambert03))
- ci\(pre-commit.ci\): autoupdate [\#24](https://github.com/tlambert03/microsim/pull/24) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- feat: use SimBaseModel and add float precision setting [\#19](https://github.com/tlambert03/microsim/pull/19) ([tlambert03](https://github.com/tlambert03))
- cosem and cleanup [\#17](https://github.com/tlambert03/microsim/pull/17) ([tlambert03](https://github.com/tlambert03))
- Smaller psf [\#16](https://github.com/tlambert03/microsim/pull/16) ([tlambert03](https://github.com/tlambert03))
- ci\(dependabot\): bump codecov/codecov-action from 3 to 4 [\#15](https://github.com/tlambert03/microsim/pull/15) ([dependabot[bot]](https://github.com/apps/dependabot))
- ci\(dependabot\): bump actions/setup-python from 4 to 5 [\#14](https://github.com/tlambert03/microsim/pull/14) ([dependabot[bot]](https://github.com/apps/dependabot))
- ci\(dependabot\): bump softprops/action-gh-release from 1 to 2 [\#13](https://github.com/tlambert03/microsim/pull/13) ([dependabot[bot]](https://github.com/apps/dependabot))
- Reorganize code [\#12](https://github.com/tlambert03/microsim/pull/12) ([tlambert03](https://github.com/tlambert03))
- Fix import and formatting issues in multiple files [\#11](https://github.com/tlambert03/microsim/pull/11) ([tlambert03](https://github.com/tlambert03))
- update precommit [\#7](https://github.com/tlambert03/microsim/pull/7) ([tlambert03](https://github.com/tlambert03))
- use mypyc for bresenham [\#6](https://github.com/tlambert03/microsim/pull/6) ([tlambert03](https://github.com/tlambert03))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
