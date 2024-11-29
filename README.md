# Tackling battery thermal fires using AI and simulation

## The problem in a couple of sentences

There was no easy way to fit Ordinary Differential Equation (ODE) models to calorimetry data obtained from battery thermal runaway testing; a critical step in enabling simulation of battery fires. This work solves that problem by using AI to create accurate, physically consistent models to predict battery thermal runaway, that can be plugged into large-scale numerical simulations.

## Who this work benefits

Anyone working with technology that requires investigation of battery thermal safety within their application: from smartphone designers to EV battery manufacturers and aircraft designers, and would like to apply modeling and simulation for their analysis. More broadly, the work will also benefit anyone looking to fit ODEs to (irregularly sampled) time series data and wants an understanding of how Neural ODEs can be applied to similar problems.

## Requirements

**The data used in the paper is confidential and cannot be shared.**

To use this code for fitting ARC (Accelerating Rate Calorimetry) data, provide a file with 3 columns: time (in seconds), temperature (in K)(check) and temperature rate (in K/s)

The libraries required to run the code are listed in the requirements.txt.


## Citation

If this work is useful to you, please cite the following:

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
Advances in electrification and energy storage have changed the way we work, communicate, and commute; and these changes are only getting more omnipresent and far-reaching. One way these changes have been effected in the world is through batteries; they serve as standardized, dense mediums of energy storage and transport.
Unfortunately, as humanity's use of batteries has got more ambitious and ubiquitous (electric planes?) the world has observed increasing incidents of serious battery fires that have destroyed the devices they were used in (think small phones to EVs and even aircraft). This phenomenon is called thermal runaway (link) and seriously affects public confidence in the technology and decreases the rate of adoption. Turns out, depending on their internal chemistries, batteries can be HIGHLY combustible. In a controlled environment, a battery's temperature during thermal runaway can spike like this:

(link Altair press release)The way around this is to introduce design variations in battery packs that reduce the risk of thermal runaway. To make this design process commercially viable, pack designers can not afford to blow up 100s of millions of dollars worth of battery packs during the design phase to ensure thermal safety; this is where simulation comes in! Battery designs can be simulated on computers instead of in the real world, saving tens of thousands of dollars per battery pack tested. However, this requires accurate thermal models of the battery in question to run in the simulation. Wouldn't it be great if we could come up with models that exactly replicate the real-world thermal behaviour of batteries? It has been tried in the past, but those methods relied on concepts like linear regression and resulted in fits that looked like this:


We can now use artificial intelligence to learn accurate battery thermal runaway models that can be used to predict if batteries will catch fire in a given design configuration under stressful conditions. The code in this repository can be used to create AI models that fit accurate, physically consistent thermal runaway models based on Arrhenius Ordinary Differential Equations (ODEs) to calorimetry data, which represents the most common way thermal runaway models for battery cells have been (attempted to be) created in past decades, minus the AI part. These models can then be plugged into broader 3D numerical simulations of the real world, allowing battery design without blowing up a single real cell!





