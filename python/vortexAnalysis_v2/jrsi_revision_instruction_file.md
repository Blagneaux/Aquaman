# VS Code modification instructions for JR Soc Interface pre-submission revision

## Purpose

Use this file as the instruction prompt for a VS Code AI coding/editing agent. The goal is to revise the manuscript and any associated analysis scripts before submission to *Journal of the Royal Society Interface*. The revision should address the weakest points identified in the deep-search review while incorporating the author clarifications below.

Do not invent data, ethics approvals, animal numbers, experimental details, or results. Where information is missing, insert a clear placeholder only if the manuscript cannot be made accurate without it. Keep all scientific claims proportional to the evidence.

---

## Author clarifications to incorporate

1. **Study species**  
   The test subjects were striped eel catfish, *Plotosus lineatus*.

2. **Video and sample dataset**  
   The dataset contains 40 videos segmented into 493 discrete samples.

3. **Retained samples used in the study**  
   The study uses:
   - 375 samples that required only minor temporal cropping.
   - 20 samples that required heavier temporal cropping but were still made usable.

   Therefore, 395 of 493 samples were retained for analysis, corresponding to approximately 80.1% of the segmented samples.

4. **Samples excluded or requiring special handling**  
   Add a transparent accounting table in Methods or Supplementary Methods:

   | Category | Number of samples | Percentage of 493 | Outcome |
   | --- | ---: | ---: | --- |
   | Minor temporal cropping only | 375 | 76.1% | Retained |
   | Heavy temporal cropping required | 20 | 4.1% | Retained |
   | Poor segmentation | 11 | 2.2% | Excluded or flagged, depending on current pipeline |
   | No detection in some frames | 26 | 5.3% | Excluded or flagged; cause not yet determined |
   | Fish not fully submerged / 3D interaction | 18 | 3.7% | Excluded from 2D simulation |
   | Fish did not pass in front of sensors | 26 | 5.3% | Excluded |
   | Circular segmentation error, tail touching nose | 16 | 3.2% | Excluded |
   | External interference, flake in water | 1 | 0.2% | Excluded |

   Explain that the retained dataset consisted of the first two categories only.

5. **Ethics statement**  
   Add the section heading, but leave the section blank for now. Do not generate or infer ethical approval information.

6. **Acknowledgements**  
   Add this exact text:

   This work was supported by a Research Grant from HFSP: https://doi.org/10.52044/HFSP.RGY00592022.pc.gr.153593

7. **Conflict of Interest**  
   Add this exact text:

   The authors declare no conflict of interest.

8. **Data Availability**  
   Add this exact text, adjusting only formatting to match the journal style:

   The code and simulation files supporting this study are available at 10.5281/zenodo.19224932 (Lagneaux B, Porcon C, OMARINI T, Sun B, Hamilton NM, Florence R, et al. How fishes use historical hydrodynamic motion cues in search and navigation tasks: measured and simulated pressure for vortex extraction and analysis. Zenodo; 2026.). Raw experimental data (pressure signals, simulation results, and tracking files) are available from the corresponding author upon reasonable request.

---

## Main revision priorities

### 1. Clarify biological and experimental unit information

Search the manuscript for the Methods section describing animal subjects, video acquisition, segmentation, and sample selection.

Add or revise text so that it clearly states:

- The species was striped eel catfish, *Plotosus lineatus*.
- The raw dataset contained 40 videos.
- These videos were segmented into 493 discrete samples.
- The final analysis used 395 samples: 375 with minor temporal cropping and 20 with heavier temporal cropping.
- The remaining samples were excluded for the specific reasons listed in the accounting table above.

Do not invent the number of individual fish. If the number of fish is not already present in the manuscript or metadata, insert a placeholder such as:

`[INSERT NUMBER OF INDIVIDUAL FISH]`

Add a reviewer-facing sentence explaining why this accounting matters, for example:

> This filtering procedure was applied before classification and simulation analyses so that all retained samples corresponded to segments suitable for 2D digital twin simulation and pressure-sensor comparison.

---

### 2. Add a dedicated subsection on segment independence and fish identity

Add a new Methods subsection titled:

**Assessment of fish-identity dependence among retained pressure samples**

This subsection should describe an analysis designed to address the concern that multiple segments from the same fish may not be independent. Implement the analysis in the relevant codebase if pressure matrices and sample metadata are available.

#### Required analysis design

Use the experimental pressure signal matrix already available in the project.

Inputs required:

- Experimental pressure matrix `X`, with rows as samples and columns as pressure-time or pressure-sensor features.
- Metadata table with at least:
  - `sample_id`
  - `fish_id`
  - `video_id`, if available
  - `segment_id`, if available
  - retained/excluded status

Restrict the fish-identity analysis to fish identities with at least 5 retained pressure samples.

Preprocessing:

1. Use the same preprocessing as the main pressure-signal classifier whenever possible.
2. Centre and scale features consistently with the main analysis.
3. Compute a POD/SVD of the retained experimental pressure matrix.
4. Retain enough POD modes to explain 99% of the variance.
5. Project each sample into this retained POD space.

#### Variance explained by fish identity

For each retained POD coefficient, estimate the fraction of variance explained by fish identity.

Preferred metric:

- Use one-way ANOVA-style variance decomposition for each coefficient:
  - `SS_between_fish / SS_total`

Also compute, if straightforward:

- ICC(1)-style estimate for fish identity as a random grouping factor.
- Median, minimum, maximum, and interquartile range of the fish-identity variance fraction across retained coefficients.
- A variance-weighted summary across POD coefficients:

`weighted_fish_identity_fraction = sum(lambda_k * fraction_k) / sum(lambda_k)`

where `lambda_k` is the variance explained by POD coefficient `k`.

#### POD-space distance test

Test whether samples from the same fish are closer to one another than samples from different fish.

Procedure:

1. Compute pairwise distances between samples in the retained 99% POD space.
2. Prefer Euclidean distances in variance-scaled POD coordinates, unless the existing analysis has a more defensible distance metric.
3. Label each pair as:
   - same-fish pair
   - different-fish pair
4. Report:
   - median same-fish distance
   - median different-fish distance
   - effect size, such as median difference or ratio
   - confidence interval by bootstrap, preferably resampling at fish level if possible
   - permutation-test p-value

Permutation test:

- Permute fish labels across samples while preserving the number of samples per fish.
- Recompute the same-fish versus different-fish distance contrast for each permutation.
- Use at least 1,000 permutations unless computation is prohibitive.
- Set and report a random seed.

#### Interpretation rules

Use the following logic in Results and Discussion:

- If fish identity explains little variance and same-fish samples are not meaningfully closer than different-fish samples, state that the retained pressure segments do not show strong fish-identity clustering in the pressure-signal space used for analysis.
- If fish identity explains non-negligible variance or same-fish samples are closer, do not ignore it. Add a sensitivity analysis using grouped train/test splits by fish identity or include fish identity as a limitation.
- Do not claim that fish identity “does not matter” unless the result strongly supports that. Use cautious language such as “we found limited evidence that fish identity dominated the retained pressure-signal space.”

#### Required output in manuscript

Add either a compact Results paragraph or a Supplementary Results subsection reporting:

- number of fish included in this analysis
- number of retained samples used
- number of POD modes retained for 99% variance
- median and range of fish-identity variance fraction across coefficients
- variance-weighted fish-identity fraction or ICC
- same-fish versus different-fish distance comparison
- permutation-test result

Add a small supplementary table or figure if feasible.

---

### 3. Strengthen statistical validation of classifier results

Search the manuscript and supplement for classifier performance reporting.

Add or implement the following wherever feasible:

1. Confidence intervals for:
   - classifier accuracy
   - F1 score
   - fish-like versus cylinder-like classification proportions
   - key feature or SHAP summaries, if reported quantitatively

2. Permutation or label-randomization controls:
   - For classifier 1: test whether real/simulated or fish/cylinder separability exceeds chance.
   - For classifier 2: test whether vortex descriptors classify fish-like versus cylinder-like samples above chance.

3. Robustness to data splitting:
   - Report whether splits are by segment, video, trial, or individual fish.
   - If current splits are segment-level only, explicitly acknowledge this.
   - If metadata permit it, add a grouped split by video or fish as a sensitivity analysis.

4. Avoid overclaiming moderate classifier performance:
   - Report uncertainty and effect size.
   - Avoid presenting classification as direct wake reconstruction.
   - Clarify that classification measures comparative consistency between measured pressure signals and simulated model classes.

---

### 4. Moderate biological claims

Search for strong language around:

- “hydrodynamic communication”
- “ecologically meaningful”
- “fish use”
- “wake realism”
- “digital twin validation”
- “proof”
- “demonstrates”

Replace overstrong claims with more defensible wording.

Preferred framing:

- The study shows that measured pressure signals are more consistent with a deforming fish digital twin than with a cylinder control.
- Fine-scale body deformation contributes to pressure signatures that persist in the wake.
- Vortex trajectory persistence, spatial organization, and geometry appear more informative than raw vortex intensity alone.
- These hydrodynamic signatures are plausibly relevant to lateral-line sensing, but behavioural use by conspecifics is not directly tested here.

Avoid implying:

- Direct proof that fish use these cues.
- Direct proof of hydrodynamic communication.
- Full validation of real 3D wake reconstruction.
- Individual identity recognition, unless specifically supported by analysis.

Suggested replacement sentence:

> Our results do not directly demonstrate behavioural use of these cues; rather, they show that fish-like body deformation leaves pressure signatures that remain detectable in sparse wall-pressure measurements and are more consistent with the deforming digital twin than with a rigid cylinder control.

---

### 5. Make 2D simulation limitations explicit

Search the Discussion and Limitations sections.

Ensure the limitations include:

- The simulations are 2D approximations of a 3D swimming animal.
- Samples where the fish was not fully submerged were excluded because they were not suitable for 2D simulation.
- Out-of-plane motion, body roll, depth changes, and free-surface interactions may affect the real pressure field.
- Streamwise deformation effects may partly capture real 3D or projection effects not explicitly represented in the 2D model.
- A future extension should use multi-view reconstruction or direct 3D tracking to validate the full wake geometry.

Use candid but not self-defeating wording.

---

### 6. Add required Royal Society-style declaration sections

Near the end of the manuscript, before References, add the following sections in the style used by the manuscript template.

Use these headings unless the template uses exact alternatives:

## Ethics

Leave this section blank for now.

## Data accessibility

The code and simulation files supporting this study are available at 10.5281/zenodo.19224932 (Lagneaux B, Porcon C, OMARINI T, Sun B, Hamilton NM, Florence R, et al. How fishes use historical hydrodynamic motion cues in search and navigation tasks: measured and simulated pressure for vortex extraction and analysis. Zenodo; 2026.). Raw experimental data (pressure signals, simulation results, and tracking files) are available from the corresponding author upon reasonable request.

## Authors' contributions

If author contributions are already available elsewhere, move them here and format them consistently. If they are not available, insert:

`[INSERT AUTHOR CONTRIBUTIONS]`

Do not invent contributions.

## Competing interests

The authors declare no conflict of interest.

## Funding

This work was supported by a Research Grant from HFSP: https://doi.org/10.52044/HFSP.RGY00592022.pc.gr.153593

## Acknowledgements

This work was supported by a Research Grant from HFSP: https://doi.org/10.52044/HFSP.RGY00592022.pc.gr.153593

If the journal template prefers funding and acknowledgements to be distinct, keep the HFSP grant under Funding and leave Acknowledgements for non-funding thanks. Do not duplicate the same sentence if this would violate the template style.

---

### 7. Improve manuscript language and readability

Perform a scientific English copy-edit.

Priorities:

- Correct non-native phrasing.
- Shorten long sentences.
- Keep terminology consistent.
- Avoid shifting between “realism”, “similarity”, “fish-like”, “ecologically meaningful”, and “communication” without definition.
- Define “fish-like” operationally as classifier consistency with the deforming fish digital twin, not as a direct behavioural or biological truth.

Examples of wording to fix if present:

- “the model that generate them” → “the model that generated them” or “the model used to generate them”
- “respond on to” → “respond to”
- “the one I ended up using” → “the samples retained for analysis”
- “almost partially out of water” → “partly out of the water”
- “more circular than fish-like” → “approximately circular rather than elongated”

---

### 8. Update abstract and conclusion after all edits

Revise the abstract last.

The abstract should contain:

1. Biological context: lateral-line sensing and hydrodynamic wakes.
2. Method: wall-pressure measurements, deforming fish digital twin, cylinder control, interpretable classifiers.
3. Main result: measured pressure signals are more consistent with the deforming fish model, especially when relevant deformation components are retained.
4. Mechanistic interpretation: persistent vortex geometry and spatial organization contribute more than raw intensity alone.
5. Limitation: this is an indirect 2D simulation-based validation, not direct proof of behavioural cue use.
6. Contribution: a framework for comparing measured pressure signals with interpretable hydrodynamic simulations.

Avoid ending the abstract with an overstrong biological claim.

---

## Expected deliverables from the VS Code agent

1. Updated manuscript text.
2. Updated supplementary text, if needed.
3. New or updated analysis script for fish-identity POD/ICC/distance analysis.
4. New result table or supplementary figure for the independence analysis, if the required data are available.
5. Updated declarations section.
6. A short changelog listing:
   - files edited
   - sections changed
   - analyses added
   - placeholders still requiring author input

---

## Acceptance checklist before stopping

Before finishing, verify that:

- [ ] The species *Plotosus lineatus* is stated in Methods.
- [ ] The 40 videos / 493 segmented samples / 395 retained samples are clearly reported.
- [ ] The sample exclusion table is present and internally consistent.
- [ ] The manuscript explains that only minor-cropping and heavy-cropping samples were retained.
- [ ] The fish-identity dependence analysis is added or clearly marked as requiring data.
- [ ] Classifier uncertainty and/or permutation controls are added where possible.
- [ ] The unit of analysis and split strategy are explicitly described.
- [ ] Claims about hydrodynamic communication are moderated.
- [ ] The 2D limitation is explicit.
- [ ] Ethics heading exists but contains no invented approval statement.
- [ ] Data accessibility, competing interests, funding, and acknowledgements are present.
- [ ] No unsupported claims or fabricated details were introduced.
- [ ] The manuscript still reads as a coherent scientific article rather than a response memo.
