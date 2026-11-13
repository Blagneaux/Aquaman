Review POD



bold = comments from the reviewer; normal = my thoughts on it



#### **reviewer 1**

* **Even though the paper deals with an often overlooked topic in flow reconstruction, demonstrating the details of breakdown of POD when applied to wall bounded and transient wakes presented well thru the paper. Overall, I like the structure of the paper, specifically the benchmark configuration clearly described. I have some points and suggestions presented below.**



This is simply is introduction



* **The simulations are 2D, but cylinder wakes undergo 3D transitions (such as Mode A and Mode B instabilities) for the studied range of Reynolds number. If 3D simulations are outside of the current computational scope, this should be explicitly stated in the Limitations and it should state how 3D spanwise decorrelation and vortex dislocation would likely exacerbate POD model truncation and further diminish wall-pressure observability. Therefore, why choosing to do 2D simulations: restricting the computation to 2D suppresses spanwise instabilities and artificially amplifies 2D coherence, which raises the question, is the observed insensitivity to Re number above 2000 a genuine physical phenomenon, or merely a numerical artifact of enforcing 2D in an inherently three dimensional turbulent flow regime?**



We did 2D because in the broader project this study takes part in, we are bounded to 2D simulations. The goal was therefore to test the tools in the context of the project, where there are also evidence of the possibility to model the wake as 2D even though it is indeed 3D. I understand though that for this paper alone, and that in the hydrodynamic field, this is an issue. I guess we could either try to answer base on pure theory and have a paragraph in the limitation that discuss how the results would change, or we could propose a targeted 3D expansion. The issue with the later, is that this whole study was done using LilyPad, which is a 2D CFD software. We could switch to using WaterLily and have a 3D benchmark for the Re sensitivity analysis, but, first I dont know how we would introduce it since it changes the setup, I dont know what type of cylinder we should aim for (infinitely tall or finite?), and I dont know if the pressure modeling of this 3D software is stable enough (I dont remember if the POD was compute on pressure or vorticity, which if it was on vorticity, would solve this issue). I think the second option is the best nevertheless.



* **In Section 3.5, a fixed seed, case wise randomized snapshot partition is utilized to ensure representation of both transient and late time phases. In continuous dynamical systems, randomly sampled time snapshots share strong temporal autocorrelation with adjacent points. If trajectory split (leave one run out) or blocked validation cannot be readily recalculated, explicitly add a sentence in the Limitations noting that random snapshot splitting may yield an optimistic lower bound on the test error due to temporal continuity.**



I agree with adding a sentence in the Limitation. I guess we could have done multiple training with leave one out runs and swap these left out runs and then average the results, but this would mean, for example for the very small gap case, that if it was the left out run, the model would train without seeing any snapshot from this category. I am not sure to understand what he means by blocked validation though, so maybe this is a good idea, or maybe not



* **Regime local gappy POD exhibits pronounced error spikes near the regime transition thrsholds. The authors should discuss whether interpolating model bases or using soft clustering/fuzzy memebership formulations across regimes could prevent these localized spikes. If this is beyond the paper's scope, state the lack of continuous basis interpolation across gap thresholds as an explicit limitation of the discrete regime classification.**



Agree to state that as a limitation. I dont think we should extend the paper even more



* **Why was a Cartesian grid applied across the wall gap without dedicated boundary layer refinement along the tank floor? At small gap ratios (h/D < 0.44), wake physics is heavily governed by the interaction between the cylinder's lower shear layer and the adverse pressure gradient separation of the wall boundary layer. If the near wall viscous sublayer and secondary wall vortex shedding are not adequately resolved, the rapid rise in POD reconstruction error in the small gap regime could be driven by unresolved wall shear dynamics rather than a fundamental representational limit of POD. The authors should clarify how the wall boundary layer was treated and verify that the grid resolution at the wall was sufficient to capture these wall induced secondary separations.**



I find this comment very interesting. First, the cartesian grid was initially selected to give equal representation to the full gap range to mimic the randomness of the gap between the fish swimming and the wall of the tank. Once again, this comes from a very specific context, and I undersand it might be the right decision for a purely hydrodynamic paper. I guess we could propose a refinement of the small gap with the addition of a few scenarios to add to the pool. Then, I completely agree we should check the resolution to be sure the small gap are adequately resolved. We should probably also have a word on the wall boundary used in LilyPad



* **In section 4.5, the authors report that reconstruction errors during transient segments are an order of magnitude lower than those across full duration stationary cases. The authors should clarify whether this lower RMSE is primarily driven by smaller fluctuation amplitudes before full vortex street development, or by higher spatial coherence of the initial shear layer roll up**



I guess this one is straightforward and we simply have to check what he is asking for and report it.



* **For the benchmark, authors should elaborate on why uniform spacing was chosen over variance weighted or QR pivot based sparse sensor placement. If testing alternative sensor arrangements is not feasible, highlight in section 5.2 that fixed uniform sensor configuration represent a performance floor, whereas optimized sparse sensor layouts mau recover improved observability.**



This once again come down to the fact that we work in a very specific context of monitoring a freely swimming fish. The uniformly spaced sensors being the decision which was made for the experimental setup, we simply choose to model a miroring setup. For the experimental setup, the decision was made because the fish is freely swimming and therefore ends up having an kind of random motion. Nevertheless, I agree we should add the performance floor precision the reviewer is aksing for.



* **To strengthen the context surrounding POD energy distribution, coherent structures, and wake reconstruction fidelity, the authors should incorporate and discuss the following relevant studies in section 1, section 2.2 and section 5.1: DOI 10.1016/j.oceaneng.2025.122584; DOI 10.1016/j.oceaneng.2020.107518; DOI 10.1007/s00348-015-2083-7**



Let's check those papers, but sure, why not



#### **Reviewer 2**

* **The manuscript addresses an interesting and practically relevant problem: the robustness of POD/gappy POD reconstruction for wall confined cylinder wakes and sparse wall pressure sensing. The comparison between global and regime local POD bases is useful, and the findings that regime local gappy POD reduces the mean pressure nRMSE from 0.860 to 0.733 while strongly improving the conditioning of the sensor to mode mapping is potentially valuable.**



This is also just is introduction



* **My main concern is how to interpret the POD sensitivity analysis. For each configuration, the POD basis is constructed from the same snapshots that are reconstructed, and the number of modes is selected independently to retain 99% of the variance. Under this procedure, differences in absolute RMSE may mainly reflect differences in fluctuation amplitude rather than differences in the intrinsic quality or robustness of the POD representation. Therefore, the conclusions regarding the dominant influence of wall proximity, as well as the effects of windowing and resolution, should be reconsidered**



First, about the snapshots being the same used to compute the POD basis and then being reconstructed. We indeed did this in the sensitivity analysis, but not in the following comparison between POD and gappy POD. The aim was to have a reconstruction benchmark, thus to study how much of the information was lost in the dimensionality reduction. I therefore dont understand what he would have expected me to do, nor what is the problem. Second, for the fixed variance, I agree this could be interesting to have a sister study which instead of fixing the variance, fix the number of modes.



* **I also recommend that the authors clarify what "energy" means for pressure and vorticity POD, provide more information on numerical verification and grid/time step independence, and better distinguish the present synthetic sparse sensing test from true experimental data fusion. The eight pressure sensors are sampled from the same CFD data used to construct the POD bases, so they do not represent measurement noise or CFD experiment discrepancy. The random snapshot train/validation/test split also provides only a limited test of generalization because snapshot from the same Re-gap cases occur in different subsets.**



I agree to clarify what we mean by energy and to provide more information about grid and time step independence, though for the later, I am not sure to understand what it means. As for the data fusion, I guess we should make it even more clear that data fusion is the end goal and that it is not done in the present study. Before actually doing data fusion, we first check if POD is worth testing, hence we only use simulated data, even for the eight sensors. So basically I agree with what he says about the data fusion, but I think this is because I wasnt clear enough that data fusion is merely a motivation to the study. Finally, for the randomized snapshot split, this sound similar to reviewer 1 comment, though here I would say that we didnt really tried to demonstrate generalization, since we acknowledge there are different regimes and have them all in the dataset. I guess I understand his issue about the random split less than I understand reviewer 1's issue about it.



* **The manuscript would be substantially strengthened by (i) reporting normalized reconstruction errors together with the number of modes required to reach a prescribed retained variance; (ii) comparing cases at a common fixed POD rank and/or using projection onto a common basis; (iii) adding a case wise or blocked validation in which complete operating conditions are excluded from training; (iv) testing sensitivity to sensor number/location and, preferably, measurement noise; and (v) additions would make it possible to separate changes in flow fluctuation amplitude from genuine deterioration of POD representability and sparse observability.**



I think this is just a summarize of the previous comments, with proposed responses. The only new thing would be testing the number of sensors, which I wonder if it is not already done. For measurement noise, I guess we could simmulate it, but I feel like it would fall out of the scope of the paper, even more if we emphazise that data fusion is just a future target. The last point is also new, but is a consequence of the previous demands he makes, so I guess we would first have to check the previous points before looking at this one.



* **The manuscript has merit, particularly the sparse observability analysis, but the POD sensitivity methodology should be revised before the main conclusion can be considered sufficiently supported. Recommendation: Major revisions**



This is just the conclusion of his message.



Overall, I agree with most of what they say, I just dont know if we should really try to implement all of their propositions. 

