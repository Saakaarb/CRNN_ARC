# Tackling battery thermal fires using AI and simulation

## The problem in a couple of sentences 

Thermal runaway leads to battery fires in EVs, and needs to be modeled accurately to predict its occurrence. However, there was no easy way to obtain thermal runaway models from battery thermal runaway experimental testing; a critical step in enabling prediction of battery fires. This work solves that problem by using AI to create accurate, physically consistent models to predict battery thermal runaway, that can predict battery fires and be plugged into large-scale numerical simulations.

## Who this work benefits

Anyone working with technology that requires investigation of battery thermal safety within their application: from smartphone designers to EV battery manufacturers and aircraft designers, and would like to apply modeling and simulation for their analysis. More broadly, the work will also benefit anyone looking to fit ODEs to (irregularly sampled) time series data and wants an understanding of how Neural ODEs can be applied to similar problems.

## Requirements

**The data used in the paper is confidential and cannot be shared.**

To use this code for fitting ARC (Accelerating Rate Calorimetry) data, provide a file with 3 columns: time (in seconds), temperature (in K) and temperature rate (in K/s)

The libraries required to run the code are listed in the requirements.txt and can be installed using: ```pip install -r requirements.txt```


## Citation

If you found this work useful, please consider citing the original article:

```
Saakaar Bhatnagar, Andrew Comerford, Zelu Xu, Davide Berti Polato, Araz Banaeizadeh, Alessandro Ferraris,
Chemical Reaction Neural Networks for fitting Accelerating Rate Calorimetry data,
Journal of Power Sources,
Volume 628,
2025,
235834,
ISSN 0378-7753,
https://doi.org/10.1016/j.jpowsour.2024.235834.
```

# So why is this interesting?
Advances in electrification and energy storage have changed the way we work, communicate, and commute; and these changes are only getting more omnipresent and far-reaching. An important medium of storage and distribution of energy are batteries; they serve as standardized, dense mediums of energy storage and transport.
Unfortunately, as humanity's use of batteries has become more ambitious and ubiquitous (electric planes?) the world has observed increasing incidents of serious battery fires that have destroyed the devices they were used in and put many lives at risk (think small phones to [EVs](https://fireisolator.com/the-risk-of-lithium-ion-batteries-thermal-runaway-in-evs/) and even [aircraft](https://en.wikipedia.org/wiki/2013_Boeing_787_Dreamliner_grounding)). This phenomenon is called [thermal runaway](https://en.wikipedia.org/wiki/Thermal_runaway) and seriously affects public confidence in the technology and decreases the rate of adoption. Turns out, depending on their internal chemistries, batteries can be HIGHLY combustible. In a controlled environment, a battery's temperature during thermal runaway can spike like this:

<p align='center'>
<img src=images/time_temp.png width="400">
</p>

One does not want to be sitting in an EV if that happens to its battery pack! The way around this is to introduce design variations in battery packs that reduce the risk of thermal runaway. To make this design process commercially viable, pack designers can not afford to blow up 100s of millions of dollars worth of battery packs during the design phase to ensure thermal safety; this is where simulation and AI come in! Battery designs can be simulated on computers instead of in the real world, saving tens of thousands of dollars per battery pack tested. However, this requires accurate thermal runaway models of the battery to run in the simulation. Wouldn't it be great if we could come up with models that exactly replicate the real-world thermal behaviour of batteries? It has been tried in the past, but those methods relied on an oversimplification of the physics and resulted in models whose predictions looked like this compared to real-world data:

<p align='center'>
<img src=images/check_temp_before_2_stage.png width="400">
 <img src=images/check_rate_before_2_stage.png width="400">
</p>

Clearly not a good result! The obtained model is not representative of the physics of battery thermal runaway. We can now use artificial intelligence to learn accurate battery thermal runaway models that can be used to predict if batteries will catch fire in a given design configuration under dangerous conditions. Unlike other methods, the proposed AI models are unique in that they ***explicitly encode the chemistry*** of the combustion of battery components, leading to highly accurate predictions of thermal runaway, since model outputs are forced to obey the laws of combustion. The model architecture is depicted in the following figures

<p align='center'>
<img src=images/crnn_stage_i_diagram.PNG width="400">
 <img src=images/temp_CRNN_workflow.PNG width="600">
</p>


The first figure shows a single submodel, and the second figure shows how these submodels are integrated into the training loop. Further details are available in the [published paper](https://www.sciencedirect.com/science/article/pii/S0378775324017865). This results in thermal runaway models that are much better at predicting the physics of thermal runaway! Shown below are the predictions from the AI models compared to real-world experimental data:

<p align='center'>
<img src=images/check_temp_after_2_stage.png width="400">
 <img src=images/check_rate_after_2_stage.png width="400">
</p>


These models can then be plugged into broader 3D numerical simulations of the real world, allowing battery design without blowing up a single real cell! The plot below shows the comparison of the simulated temperature (in a 3D simulation) of a thermal runaway test of a battery to experimental data from a thermocouple; the simulation shows excellent agreement with real-world data.

<p align='center'>
<img src=images/heated_cell.png width="200">
 <img src=images/4_stage_acusolve.png width="600">
</p>






