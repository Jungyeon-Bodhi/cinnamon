#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri March 14 15:22:04 2025

@author: Bodhi Global Analysis (Jungyeon Lee)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.image as mpimg
from PIL import Image

bodhi_blue = (0.0745, 0.220, 0.396)
bodhi_grey = (0.247, 0.29, 0.322)
bodhi_primary_1 = (0.239, 0.38, 0.553)
bodhi_secondary = (0.133, 0.098, 0.42)
bodhi_tertiary = (0.047, 0.396, 0.298)
bodhi_complement = (0.604, 0.396, 0.071)
color_palette = [bodhi_primary_1, bodhi_complement, bodhi_tertiary, bodhi_blue, bodhi_grey, bodhi_secondary]


def visual(data, title, fontsize, output_file):
    subgroup_colors = {
        "ASEAN": bodhi_blue,
        "CML": bodhi_primary_1,
        "High Income": bodhi_secondary,
        "Island Group": bodhi_tertiary,
        "VI": bodhi_complement
    }
    
    categories = ["Economic", "Environmental", "Human", "Societal"]
    x = np.arange(len(categories))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, subgroup in enumerate(data["Sub-group"]):
        values = [data[cat][i] for cat in categories]
        color = subgroup_colors.get(subgroup, "gray")
        ax.bar(x + i * width, values, width, label=subgroup, color=color)
        
        for j, v in enumerate(values):
            ax.text(x[j] + i * width, v + 0.5, f'{v:.1f}', ha='center', fontsize=fontsize)
    
    ax.set_xticks(x + width)
    ax.set_ylim(0, 103)
    ax.set_xticklabels(categories)
    ax.set_ylabel(" ")
    ax.set_title(title, fontsize=fontsize+2)
    ax.legend(title="ASEAN Group", fontsize=fontsize)
    ax.axhline(y=30, color='lightgreen', linestyle='--', linewidth=0.6)
    ax.axhline(y=70, color='orange', linestyle='--', linewidth=0.6)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=600)
    plt.close()
    
def visual1(data, year, title, fontsize, output_file, ylim=None, padding = None):
    df = pd.DataFrame(data)
    
    colors = cm.tab10(range(len(df.index)))
    
    plt.figure(figsize=(11, 8))
    
    if padding != None:
        padding_text = padding
    else: padding_text = 0.05

    for i, country in enumerate(df['Country']):
        country_data = df.iloc[i, 1:].values 
        years = df.columns[1:]  
    

        valid_years = []
        valid_data = []
        
        for j, value in enumerate(country_data):
            if not np.isnan(value):
                valid_years.append(years[j])
                valid_data.append(value)
        

        plt.plot(valid_years, valid_data, marker='o', label=country, color=colors[i], markersize=5)
    

        for j, value in enumerate(valid_data):
            plt.text(valid_years[j], value + padding_text, f'{value}', ha='center', va='bottom', color=colors[i], fontweight='bold', fontsize = 8.5)
    

    plt.xlabel(" ")
    plt.ylabel(" ")
    if ylim != None:
        plt.ylim(ylim[0],ylim[1])
        range_y = np.arange(ylim[0],ylim[1],ylim[2])
        plt.yticks(range_y)
    plt.xticks(year)

    plt.title(title, fontsize= fontsize+4)
    plt.legend(title="Country", bbox_to_anchor=(1.125, 1), loc='upper right', fontsize = 8, borderpad=0.3, labelspacing=0.3)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=600)
    plt.close()
    
def visual2(data, year, title, fontsize, output_file, ylim=None, padding=None):
    df = pd.DataFrame(data)
    
    num_countries = len(df["Country"])
    df[year] = df[year].fillna(0)
    num = len(year)
    half = int(num_countries // 2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharey=True)  # 2 rows, 1 column, shared y-axis
    pil_img = Image.open('asean_map.png')  # Adjust path if necessary
    width, height = pil_img.size
    pil_img_resized = pil_img.resize((int(width // 1.6), int(height // 1.6)))  # Resize to half size

    img_resized = np.array(pil_img_resized)
    
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)  # Make the background fully transparent
    bar_width = 0.17
    
    index1 = np.arange(half) 
    index2 = np.arange(half) + half 
    bodhi_blue = np.array([0.0745, 0.220, 0.396])
    colors = [bodhi_blue + (i * 0.1) for i in range(len(year))]

    bars1 = []
    for i, year in enumerate(year):
        bars = ax1.bar(index1 + i * bar_width, df[year][:half], bar_width, label=str(year), color=colors[i])
        bars1.append(bars)
        for bar in bars:
            bar_height = bar.get_height()
            text_color = 'white' if bar_height < 0 else 'black'  # Set text color based on the bar's height
            ax1.text(bar.get_x() + bar.get_width() / 2, bar_height, f'{bar_height:.2f}', 
                     ha='center', va='bottom', fontsize=fontsize, color=text_color)
    
    bars2 = []
    for i, year in enumerate(year):
        bars = ax2.bar(index2 + i * bar_width, df[year][half:], bar_width, label=str(year), color=colors[i])
        bars2.append(bars)
        for bar in bars:
            bar_height = bar.get_height()
            text_color = 'white' if bar_height < 0 else 'black'  # Set text color based on the bar's height
            ax2.text(bar.get_x() + bar.get_width() / 2, bar_height, f'{bar_height:.2f}', 
                     ha='center', va='bottom', fontsize=fontsize, color=text_color)
    
    xticks = np.arange(num_countries)

    if ylim != None:
        ax1.set_ylim(ylim[0], ylim[1])
        ax2.set_ylim(ylim[0], ylim[1])
        range_y = np.arange(ylim[0],ylim[1],ylim[2])
        ax1.set_yticks(range_y)
        ax2.set_yticks(range_y)
    
    ax1.set_xticks(index1 + bar_width * 2)
    ax1.set_xticklabels(df["Country"][:half])
    
    ax2.set_xticks(index2 + bar_width * 2) 
    ax2.set_xticklabels(df["Country"][half:]) 
    
    ax1.set_title(f"{title}", fontsize = fontsize + 3)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_title('')

    fig.patch.set_facecolor('white') 
    fig.subplots_adjust(hspace=0.0)   
    
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.figimage(img_resized, xo=300, yo=75, alpha=0.1, zorder=0)
    
    handles, labels = [], []
    for i, year in enumerate(year[:num]): 
        bars = bars1[i] + bars2[i] 
        handles.append(bars[0]) 
        labels.append(str(year))
    
    fig.legend(handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, 0.45), 
               ncol=5, fontsize=7)

    
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, transparent=True, bbox_inches='tight')
    plt.show()

def visual3(data, years, title, fontsize, output_file, ylim=None, padding=None):
    df = pd.DataFrame(data)
    num_countries = len(df["Country"])
    df[years] = df[years].fillna(0)
    num = len(years)
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    pil_img = Image.open('asean_map.png')
    pil_img_resized = pil_img.resize((int(pil_img.width // 1.6), int(pil_img.height // 1.6)))
    img_resized = np.array(pil_img_resized)
    
    fig.patch.set_alpha(0.0)
    bar_width = 0.19
    index1 = np.arange(num_countries)
    
    bodhi_blue = np.array([0.0745, 0.220, 0.396])
    colors = [np.clip(bodhi_blue + (i * 0.1), 0, 1) for i in range(num)]
    
    bars1 = []
    for i, year in enumerate(years):
        bars = ax1.bar(index1 + i * bar_width, df[year], bar_width, label=str(year), color=colors[i])
        bars1.append(bars)
        for bar in bars:
            bar_height = bar.get_height()
            text_color = 'white' if bar_height < 0 else 'black'
            ax1.text(bar.get_x() + bar.get_width() / 2, bar_height, f'{bar_height}', 
                     ha='center', va='bottom', fontsize=8, color=text_color)
    
    ax1.set_xticks(index1 + bar_width * (num / 2))
    ax1.set_xticklabels(df["Country"], fontsize= fontsize)
    
    if ylim is not None:
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.set_yticks(np.arange(ylim[0], ylim[1] + ylim[2], ylim[2]))
    
    ax1.set_title(title, fontsize=fontsize + 8)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    
    fig.figimage(img_resized, xo=340, yo=120, alpha=0.1, zorder=0)
    
    fig.legend(handles=[bars1[i][0] for i in range(num)], labels=[str(year) for year in years],
               loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=min(num, 3), fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, transparent=True, bbox_inches='tight')
    plt.show()

def economic_dimension():
    title1 = "Economic Indicator: Exchange rate (%)"
    years = [2019, 2020, 2021, 2022, 2023]
    data1 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2019 : [1.13, 0.25, -0.63, 3.31, 2.66, 6.19, -1.64, 1.14, -3.91, 1.98],
        2020 : [1.14, 0.78, 3.07, 4.22, 1.47, -9.00, -4.19, 1.14, 0.79, 0.69],
        2021 : [-2.60, 0.15, -1.88, 7.21, -1.43, 16.92, -0.74, -2.63, 2.18, -0.21],
        2022 : [2.62, 0.08, 3.79, 44.72, 6.22, 19.63, 10.60, 2.62, 9.65, 0.48],
        2023 : [-2.60, 0.21, 2.61, 26.03, 3.63, 8.67, 2.12, -2.60, -0.74, 2.22]}
    
    visual2(data1, years, title1, fontsize = 9, output_file = 'visuals/eco_exchange_rate.png', ylim=(-12, 50, 5), padding = 0.1)
    
    title2 = 'Economic Indicator: Financial inclusion'
    years = [2019, 2020, 2021, 2022, 2023]
    data2 = {
    "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
    2019 : [17.6, 8.3, 15.7, 3.1, 9.8, 5.6, 9.2, 7.8, 10.9, 4.0],
    2020 : [17.4, 11.6, 15.3, 3.1, 8.7, np.nan, 9.2, 7.0, 10.3, 4.0],
    2021 : [16.2, 12.4, 15.9, 3.1, 8.5, np.nan, 9.0, 6.9, 9.4, 2.9],
    2022 : [np.nan, 12.6, 12.4, np.nan, 8.2, np.nan, np.nan, 6.4, 8.7, 3.0],
    2023 : [np.nan, 14.3, 11.7, np.nan, 8.1, np.nan, np.nan, 6.1, 8.3, 3.1] }

    visual2(data2, years, title2, fontsize = 9, output_file = 'visuals/eco_financial_inclusion.png')

    title3 = 'Economic Indicator: GDP per capita growth'
    years = [2019, 2020, 2021, 2022, 2023]
    data3 = {
    "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
    2019 : [2.7, 6.6, 4.0, 3.9, 2.8, 5.8, 4.8, 0.2, 1.9, 6.3],
    2020 : [0.1, -5.0, -2.9, -1.0, -6.7, -9.7, -10.5, -3.6, -6.2, 1.9],
    2021 : [-2.5, 1.6, 3.0, 1.1, 2.1, -12.6, 4.8, 14.4, 1.4, 1.7],
    2022 : [-2.4, 3.7, 4.5, 1.3, 7.6, 3.3, 6.8, 0.5, 2.5, 7.3],
    2023 : [0.6, 3.6, 4.2, 2.3, 2.3, 0.3, 4.7, -3.7, 1.9, 4.3] }

    visual2(data3, years, title3, fontsize = 9, output_file = 'visuals/eco_gdp_capita.png', ylim=(-13, 15, 1))

    title4 = 'Economic Indicator: Estimated real GDP growth (Annual percent change)'
    years = [2025, 2026, 2027, 2028, 2029]
    data4 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2025 : [2.5, 5.8, 5.1, 3.5, 4.4, 1.1, 6.1, 2.5, 3, 6.1],
        2026 : [2.6, 6.2, 5.1, 3.1, 4.4, 1.3, 6.3, 2.5, 2.6, 6],
        2027 : [2.7, 6.1, 5.1, 2.8, 4, 1.5, 6.3, 2.5, 2.7, 5.8],
        2028 : [2.9, 6, 5.1, 2.6, 4, 1.6, 6.3, 2.5, 2.7, 5.8],
        2029 : [3.1, 6, 5.1, 2.5, 4, 1.8, 6.3, 2.5, 2.7, 5.6] }
    
    visual2(data4, years, title4, fontsize = 9, output_file = 'visuals/eco_gdp_capita(est).png', ylim=(0.5, 7, 0.5))
    
    title5 = 'Economic Indicator: Natural resource dependence (%)'
    years = [2017, 2018, 2019, 2020, 2021]
    data5 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2017 : [17.2, 1.9, 3.1, 5.0, 6.2, 7.8, 1.0, 0.0, 1.6, 3.2],
        2018 : [22.5, 0.9, 4.2, 3.9, 7.3, 6.7, 0.9, 0.0, 1.8, 3.3],
        2019 : [20.8, 0.7, 3.2, 3.1, 6.2, 5.9, 0.8, 0.0, 1.5, 2.5],
        2020 : [13.8, 0.9, 2.4, 2.8, 4.5, 4.7, 0.8, 0.0, 1.1, 1.8],
        2021 : [24.3, 0.8, 5.2, 5.4, 6.9, 8.7, 2.0, 0.0, 1.8, 2.5] }
    
    visual2(data5, years, title5, fontsize = 9, output_file = 'visuals/eco_natrual_resources.png', ylim=(0, 25, 1))
    
    title6 = 'Economic Indicator: Tax revenue'
    years = [2018, 2019, 2020, 2021, 2022]
    data6 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2018 : [18.3, 17.1, 10.3, 12.8, 12.0, 3.0, 14.1, 13.0, 16.5, 15.0],
        2019 : [12.0, 19.7, 9.8, 10.9, 11.9, 6.4, 14.5, 13.2, 16.1, 14.4],
        2020 : [5.3, 17.9, 8.3, 9.4, 10.9, np.nan, 14.0, 12.8, 15.7, 13.2],
        2021 : [9.3, 16.4, 9.1, 10.3, 11.2, np.nan, 14.1, 13.1, 15.6, 13.9],
        2022 : [12.1, np.nan, 10.4, 10.6, 11.7, np.nan, np.nan, 11.5, np.nan, np.nan]}
    
    visual2(data6, years, title6, fontsize = 9, output_file = 'visuals/eco_taxrevenue.png', ylim=(3, 22, 1))
    
    title7 = 'Economic Indicator: Unemployment rate (Aged 25 +)'
    years = [2021, 2022, 2023, 2024, 2025]
    data7 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2021 : [3.3, 0.3, 2.1, 1.7, 3.2, 2.3, 2.5, 4.0, 0.7, 1.8],
        2022 : [3.5, 0.1, 1.7, 0.9, 2.4, 1.4, 1.9, 3.1, 0.5, 1.0],
        2023 : [3.5, 0.1, 1.7, 0.9, 2.4, 1.4, 1.4, 2.9, 0.4, 1.1],
        2024 : [3.5, 0.1, 1.7, 0.9, 2.4, 1.4, 1.4, 2.7, 0.4, 0.8],
        2025 : [3.5, 0.2, 1.7, 1.0, 2.4, 1.4, 1.4, 2.6, 0.3, 0.7]}
    
    visual2(data7, years, title7, fontsize = 9, output_file = 'visuals/eco_unemployment.png', ylim=(-1, 5, 1))
    
    title8 = "Economic Indicator: Women's employment in non-agricultural sectors (%)"
    years = [2015, 2016, 2017, 2018, 2019]
    data8 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2015 : [41.8, 46.4, 38.3, 46.3, 39.9, 42.9, 45.4, 41.7, 47.3, 46.6],
        2016 : [42.2, 46.2, 39.5, 46.4, 40, 43.8, 44.4, 41.8, 47.5, 46.5],
        2017 : [42.7, 46.7, 40, 46.4, 39.9, 44.6, 43.3, 41.6, 47.5, 46.8],
        2018 : [42.7, 46.8, 40.5, 46.5, 39.9, 44.5, 43.7, 41.8, 47.6, 46.9],
        2019 : [42.8, 46.8, 40.5, 46.6, 39.9, 44.4, 43.6, 41.8, 47.6, 46.9]}
    
    visual2(data8, years, title8, fontsize = 9, output_file = 'visuals/eco_women_agri.png', ylim=(38, 50, 1))

def environment_dimension():
    title2 = 'Environmental Indicator: Air quality'
    years = [2020, 2022, 2024]
    data2 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2020 : [68.4, 30.1, 26.8, 27.8, 50.3, 20.7, 30.4, 76.9, 40.6, 32.0],
        2022 : [61.7, 25.9, 21.5, 22.6, 43.7, 16.9, 25.9, 69.2, 34.4, 26.5],
        2024 : [68.6, 18.1, 22.7, 13.6, 43.2, 9.0, 22.7, 53.7, 25.4, 15.4] }
    
    visual2(data2, years, title2, fontsize = 9, output_file = 'visuals/env_air_quality.png', ylim=(5, 80, 10))
    
    title3 = 'Environmental Indicator: Environment-related Displacement'
    years = [2019, 2020, 2021, 2022, 2023]
    data3 = {
        "Country": ["Brunei","Cambodia", "Laos", "Malaysia", 'Singapore',"Thailand"],
        2019 : [0, 70000, 103000, 63000, 0, 61000],
        2020 : [0, 66000, 12000, 23000, 0, 13000],
        2021 : [0, 15000, 5, 129000, 0, 9400],
        2022 : [0, 2800, 560, 156000,  0, 22000],
        2023 : [0, 46000, 1100, 206000, 0, 2800]}
    
    visual3(data3, years, title3, fontsize = 9, output_file = 'visuals/env_displacement.png')
    
    title3_1 = 'Environmental Indicator: Environment-related Displacement'
    years = [2019, 2020, 2021, 2022, 2023]
    data3_1 = {
        "Country": [ "Indonesia", "Myanmar","Philippines", "Vietnam"],
        2019 : [463000, 270000, 4500000, 89000],
        2020 : [705000, 50000, 4500000,  1300000],
        2021 : [749000, 158000, 5700000, 780000],
        2022 : [308000, 13000, 5500000,  353000],
        2023 : [238000, 995000, 2600000,  68000]}
    
    visual3(data3_1, years, title3_1, fontsize = 9, output_file = 'visuals/env_displacement2.png')
    
    title4 = 'Environmental Indicator: Exposure to hazards'
    years = [2021, 2022, 2023, 2024, 2025]
    data4 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2021 : [2.6, 4.7, 7.4, 3.7, 4.4, 7.2, 8.3, 1.0, 5.8, 7.4],
        2022 : [2.6, 4.6, 7.4, 3.7, 4.4, 7.2, 8.3, 1.0, 5.8, 7.4],
        2023 : [2.6, 4.6, 7.4, 3.7, 4.5, 7.2, 8.3, 1.1, 5.8, 7.4], 
        2024 : [2.6, 4.6, 7.3, 3.7, 4.5, 7.2, 8.3, 1.1, 5.9, 7.3],
        2025 : [2.6, 4.6, 7.4, 3.6, 4.5, 7.2, 8.3, 1.2, 5.9, 7.3]}
    
    visual2(data4, years, title4, fontsize = 9, output_file = 'visuals/env_exp_hazards.png', ylim=(0, 10, 1))
    
    title5 = 'Environmental Indicator: Food supply adequacy (%)'
    years = [2020, 2021, 2022, 2023]
    data5 = {
        "Country": ["Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", "Thailand","Vietnam"],
        2020 : [122.0, 126.0, 116.0, 124.0, 120.0, 124.0, 115.0, 129.0],
        2021 : [122.0, 126.0, 116.0, 124.0, 120.0, 125.0, 116.0, 131.0],
        2022 : [122.0, 124.0, 115.0, 125.0, 121.0, 125.0, 117.0, 132.0], 
        2023 : [122.0, 123.0, 115.0, 125.0, 121.0, 125.0, 118.0, 133.0]}
    
    visual2(data5, years, title5, fontsize = 9, output_file = 'visuals/env_food_supply.png', ylim=(110, 135, 5))
    
    title6 = 'Environmental Indicator: Lack of adaptive capacity'
    years = [2018, 2019, 2020, 2021, 2022]
    data6 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2018 : [0.584, 0.701, 0.605, 0.715, 0.450, 0.624, 0.568, 0.319, 0.429, 0.491],
        2019 : [0.584, 0.696, 0.527, 0.709, 0.430, 0.634, 0.546, 0.321, 0.428, 0.489],
        2020 : [0.584, 0.692, 0.522, 0.709, 0.413, 0.646, 0.531, 0.311, 0.416, 0.498], 
        2021 : [0.580, 0.693, 0.511, 0.709, 0.405, 0.658, 0.503, 0.309, 0.413, 0.482],
        2022 : [0.580, 0.683, 0.510, 0.710, 0.408, 0.657, 0.501, 0.306, 0.410, 0.481]}
    
    visual2(data6, years, title6, fontsize = 9, output_file = 'visuals/env_adaptive_capacity.png', ylim=(0.3, 0.75, 0.05), padding = 0.005)
    
    title7 = 'Environmental Indicator: Non-renewable resource crimes'
    years = [2021, 2023]
    data7 = {
        "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
        2021 : [2.0, 8.0, 7.5, 1.5, 5.5, 6.5, 8.0, 1.5, 6.0, 6.0],
        2023 : [2.0, 8.0, 8.0, 3.5, 5.5, 9.0, 8.0, 2.5, 6.0, 6.0]}
    
    visual2(data7, years, title7, fontsize = 9, output_file = 'visuals/env_non_renewable_crime.png', ylim=(1, 10, 1), padding = 0.01)
    
def human_dimension():    
    title2 = 'Human Indicator: Access to basic water (%)'
    years = [2018, 2019, 2020, 2021, 2022]
    data2 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2018 : [99.9, 74.4, 91.0, 83.3, 97.1, 77.1, 93.2, 100, 99.5, 95.2],
            2019 : [99.9, 75.3, 91.8, 85.1, 97.1, 78.9, 93.6, 100, 99.9,  96.0],
            2020 : [99.9, 76.3, 92.6, 85.2, 97.1, 80.7, 94.1, 100, 100, 96.7],
            2021 : [99.9, 77.1, 93.4, 85.3, 97.1, 82.3, 94.5, 100, 100, 97.3],
            2022 : [99.9, 78.1, 94.1, 85.5, 97.2, 82.4, 94.9, 100, 100, 98.0] }
    visual2(data2, years, title2, fontsize = 9, output_file = 'visuals/human_acc_water.png', ylim=(70, 103, 5))
    
    title3 = 'Human Indicator: Access to immunisation services (%)'
    years = [2019, 2020, 2021, 2022, 2023]
    data3 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2019 : [99, 88, 85, 80, 98, 90, 84, 97, 97, 89],
            2020 : [99, 87, 77, 79, 98, 84, 83, 98, 97, 94],
            2021 : [99, 87, 67, 75, 95, 37, 69, 98, 92, 83],
            2022 : [99, 85, 91, 80, 97, 71, 84, 97, 92, 91],
            2023 : [99, 85, 83, 84, 97, 76, 89, 98, 92, 65] }
    visual2(data3, years, title3, fontsize = 9, output_file = 'visuals/human_acc_immune.png', ylim=(35, 104, 5))
    
    title4 = 'Human Indicator: Age dependency ratio (%)'
    years = [2019, 2020, 2021, 2022, 2023]
    data4 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2019 : [37.4, 57.7, 48.3, 56.5, 44.8, 46.4, 55.7, 30.2, 40.6, 46.6],
            2020 : [37.4, 57.5, 48.0, 55.9, 44.3, 46.3, 54.8, 31.2, 41.0, 47.0],
            2021 : [37.5, 57.2, 47.6, 55.3, 43.6, 46.2, 53.6, 32.3, 41.5, 47.2],
            2022 : [37.7, 56.8, 47.3, 54.6, 42.9, 46.2, 52.4, 32.8, 41.9, 47.4],
            2023 : [38.0, 56.4, 47.0, 54.1, 42.3, 46.2, 51.2, 33.1, 42.5, 47.6] }
    visual2(data4, years, title4, fontsize = 9, output_file = 'visuals/human_age_dependency.png', ylim=(30, 60, 3))
    
    title5 = 'Human Indicator: Human inequality'
    years = [2018, 2019, 2020, 2021, 2022]
    data5 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2018 : [11.51, 26.60, 18.68, 25.09, 14.25, 22.33, 16.90, 12.81, 15.91, 17.12],
            2019 : [11.53, 26.46, 18.28, 25.56, 14.09, 22.17, 16.81, 12.66, 15.24, 17.09],
            2020 : [11.53, 26.33, 17.55, 25.33, 14.09, 21.96, 16.69, 12.58, 15.19, 16.52],
            2021 : [11.52, 26.43, 17.52, 25.17, 14.06, 21.88, 16.01, 12.53, 15.14, 16.48],
            2022 : [11.53, 26.40, 17.30, 24.74, 13.84, 21.75, 16.69, 12.55, 14.91, 16.29] }
    visual2(data5, years, title5, fontsize = 9, output_file = 'visuals/human_inequality.png', ylim=(10, 27, 3))
    
    title6 = 'Human Indicator: Primary completion rate (%)'
    years = [2019, 2020, 2021, 2022, 2023]
    data6 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2019 : [94.3, 91.9, 104.6, 93.5, 93.7, np.nan, 101.0, 98.1, 101.1, 109.6],
            2020 : [100.4, 94.9, 103.1, 91.0, 100.3, np.nan, 98.1, 101.4, 102.7, np.nan],
            2021 : [np.nan, 94.7, 103.4, 90.2, 101.1, np.nan, 91.1, 100.6, 103.5, np.nan],
            2022 : [np.nan, 90.7, 103.0, 89.1, 95.1, np.nan, 88.0, 99.6, 102.4, 115.9],
            2023 : [98.2, 89.9, 101.9, 90.0, 98.8, np.nan, 80.7, np.nan, 103.0, np.nan] }
    visual2(data6, years, title6, fontsize = 9, output_file = 'visuals/human_primary_comp.png', ylim=(80, 117, 3))
    
    title7 = 'Human Indicator: Social protection coverage'
    years = [2019, 2020, 2021, 2022, 2023]
    data7 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2019 : [34.1, 6.2, 27.8, 12.1, 27.3, 6.3, 36.7, 100, 68, 38.8],
            2020 : [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 100, 70.1, np.nan],
            2021 : [36, np.nan, 54.3, 15.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            2022 : [np.nan, 20.8, np.nan, np.nan, np.nan, np.nan, 34.9, np.nan, np.nan, 38.3],
            2023 : [np.nan, np.nan, np.nan, np.nan, 29.2, np.nan, np.nan, 100, np.nan,np.nan] }
    visual2(data7, years, title7, fontsize = 9, output_file = 'visuals/human_social_protection.png', ylim=(0, 103, 10))
    
    title8 = 'Human Indicator: Youth not in education, employment, or training (%)'
    years = [2019, 2020, 2021, 2022, 2023]
    data8 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2019 : [20.1, 11.4, 20.6, 24.6, 11.4, 13.8, 18.8, 4.1, 14.9, 13.2],
            2020 : [21.4, 5.7, 21.8, 24.6, 13.6, 15.0, 21.1, 4.5, 15.1, 14.2],
            2021 : [18.7, 6.2, 22.5, 23.4, 9.3, 16.1, 17.5, 7.5, 14.8, 14.9],
            2022 : [20.0, 12.4, 22.3, 22.5, 10.2, 15.3, 12.8, 6.6, 13.3, 11.0],
            2023 : [16.5, 12.5, 21.4, 22.4, 10.2, 15.2, 12.9, 6.8, 12.5, 10.8] }
    visual2(data8, years, title8, fontsize = 9, output_file = 'visuals/human_youth_in_education.png', ylim=(0, 27, 3))

def societal_dimension(): 
    title1 = 'Societal Indicator: Income inequality'
    years = [2018, 2019, 2020, 2021, 2022]
    data1 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2018 : [1.00, 2.70, 1.90, np.nan, 1.96, 1.44, 2.09, np.nan, 1.75, 1.54],
            2019 : [np.nan, np.nan, 1.84, 1.94, 1.98, np.nan, np.nan, np.nan, 1.65, np.nan],
            2020 : [np.nan, np.nan, 1.85, np.nan, np.nan, np.nan, np.nan, np.nan, 1.65, 1.64],
            2021: [np.nan, np.nan, 1.87, np.nan, np.nan, np.nan, 1.92, np.nan, 1.66, np.nan],
            2022: [np.nan, np.nan, 1.88, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}
    visual2(data1, years, title1, fontsize = 9, output_file = 'visuals/societal_income_inequaility.png', ylim=(0.9, 3, 0.2), padding = 0.009)
    
    title2 = 'Societal Indicator: Interpersonal trust'
    years = [2018, 2019, 2020, 2021]
    data2 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2018 : [np.nan, 0.188, 0.269, 0.277, 0.110, 0.263, 0.110, 0.182, 0.468, 0.575],
            2019 : [np.nan, 0.188, 0.069, 0.277, 0.261, 0.263, 0.110, 0.182, 0.418, 0.575],
            2020 : [np.nan, 0.188, 0.069, 0.277, 0.261, 0.263, 0.071, 0.182, 0.418, 0.575],
            2021 : [np.nan, 0.188, 0.069, 0.277, 0.261, 0.201, 0.071, 0.453, 0.418, 0.369]}
    visual2(data2, years, title2, fontsize = 9, output_file = 'visuals/societal_inter_trust.png', ylim=(0, 0.65, 0.02))
    
    title3 = 'Societal Indicator: Urbanisation rate'
    years = [2019, 2020, 2021, 2022, 2023]
    data3 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2019 : [1.5, 3.0, 2.1, 3.3, 2.3, 1.6, 1.7, 1.1, 1.7, 2.9],
            2020 : [1.5, 3.2, 2.0, 3.3, 2.1, 1.6, 1.7, -0.3, 1.6, 2.9],
            2021 : [1.3, 3.3, 1.8, 3.2, 1.8, 1.7, 1.5, -4.2, 1.5, 2.8],
            2022 : [1.2, 3.1, 1.9, 3.2, 1.9, 1.7, 1.4, 3.3, 1.4, 2.6],
            2023 : [1.2, 3.1, 1.9, 3.1, 1.9, 1.8, 1.5, 4.9, 1.3, 2.5]}
    visual2(data3, years, title3, fontsize = 9, output_file = 'visuals/societal_urban_rate.png', ylim=(-4.5, 5.2, 0.3))
    
    title4 = 'Societal Indicator: Participatory environment for CSOs'
    years = [2020, 2021, 2022, 2023, 2024]
    data4 = {
            "Country": ["Brunei","Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar","Philippines", 'Singapore',"Thailand","Vietnam"],
            2020 : [np.nan, 0.805, 2.518, -1.303, 1.265, 1.253, 1.378, -0.252, 0.764, 1.055],
            2021 : [np.nan, 0.691, 2.518, -1.111, 1.265, -0.606,1.378, -0.252, 0.764, 0.947],
            2022 : [np.nan, 0.691, 2.518, -1.332, 1.265, -1.025,1.378, -0.252, 0.784, 0.947],
            2023 : [np.nan, 0.691, 1.709, -1.332, 1.265, -1.025,1.378, -0.252, 0.784, 0.947],
            2024 : [np.nan, 0.691, 1.709, -1.332, 1.265, -1.025,1.378, -0.252, 1.015, 0.947]}
    visual2(data4, years, title4, fontsize = 9, output_file = 'visuals/societal_cso_partici.png', ylim=(-1.5, 3, 0.3), padding = 0.002)
    
    
    title5 = 'Societal Indicator: Forced displacement (Low risk group)' 
    years = [2017, 2019, 2020, 2021, 2022, 2023]
    data5 = {
            "Country": ["Brunei","Cambodia", "Laos", "Malaysia", 'Singapore',"Thailand"],
            2017 : [94, 15000, 190, 82000,  0, 50000],
            2019 : [0, 70000,  103000, 63000,  0, 61000],
            2020 : [0, 66000, 12000, 23000,  0, 13000],
            2021 : [0, 15000,  5, 129000,  0, 9920],
            2022 : [0, 28000,  560, 156000,  0, 22000],
            2023 : [0, 46000, 1100, 206000,  0, 2800],}
    visual3(data5, years, title5, fontsize = 9, output_file = 'visuals/societal_displacement1.png', ylim=(0, 210000, 10000))
    
    title6 = 'Societal Indicator: Forced displacement (High risk group)'
    years = [2017, 2019, 2020, 2021, 2022, 2023]
    data6 = {
            "Country": ["Indonesia", "Myanmar","Philippines" ,"Vietnam"],
            2017 : [377690,  407967, 3173811, 633000],
            2019 : [486396,  349652, 4678273, 89000],
            2020 : [709599,  120194, 4560622,  1264000],
            2021 : [75580,  605818, 5854145,  780000],
            2022 : [315092,  1019395, 5575507,   353000],
            2023 : [240176,  2292900, 2755046,  68000],}
    visual3(data6, years, title6, fontsize = 9, output_file = 'visuals/societal_displacement2.png', ylim=(60000, 6000000, 200000))

title1 = 'Multidimensional Fragility Framework - ASEAN'
data1 = {
    "Sub-group": ["ASEAN","CML", "High Income", "Island Group", "VI"],
    "Economic" : [39.8, 39.5, 37.1, 40.9, 40.7],
    "Environmental" : [77.2, 64.6, 51.4, 80.7, 77.5],
    "Human" : [26.1, 25.9, 13.3, 29.7, 27.0],
    "Societal" : [57.9, 60.5, 42.4, 51.7, 52.9]}

visual(data1, title1, fontsize = 9, output_file = 'visuals/mff_asean.png')

economic_dimension()
environment_dimension()  
human_dimension()
societal_dimension()