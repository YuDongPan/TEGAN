# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/11/6 13:32
import pyecharts.options as opts
from pyecharts.charts import Radar

color = '#ffffff'

'''Result data for Direction'''
v1 = [[40.27, 67.92, 37.25, 51.02]]
v2 = [[75.70, 76.61, 76.06, 76.55]]

v3 = [[61.47, 83.06, 49.69, 61.10]]
v4 = [[86.01, 86.07, 84.67, 86.07]]

'''Result data for Dial'''
# v1 = [[42.86, 65.06, 35.63, 55.79]]
# v2 = [[73.24, 72.48, 70.96, 73.32]]
#
# v3 = [[67.61, 82.93, 55.28, 69.50]]
# v4 = [[84.61, 82.44, 83.54, 84.15]]


(
    Radar(init_opts=opts.InitOpts(width="1280px", height="720px", bg_color=color))
        .add_schema(
        shape='circle',
        schema=[
            opts.RadarIndicatorItem(name="ITCCA", max_=100.00),
            opts.RadarIndicatorItem(name="TRCA",  max_=100.00),
            opts.RadarIndicatorItem(name="EEGNet", max_=100.00),
            opts.RadarIndicatorItem(name="C-CNN", max_=100.00),
        ],
        center=["50%", "50%"],
        radius="80%",
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
        ),
        textstyle_opts=opts.TextStyleOpts(color="#000000", font_size=15, font_weight='bold',
                                          font_family='Time New Roman'),
        angleaxis_opts=opts.AngleAxisOpts(
            min_=0,
            max_=360,
            is_clockwise=False,
            interval=5,
            axistick_opts=opts.AxisTickOpts(is_show=False),
            axislabel_opts=opts.LabelOpts(is_show=False),
            axisline_opts=opts.AxisLineOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        radiusaxis_opts=opts.RadiusAxisOpts(
            min_=0.00,
            max_=100.00,
            interval=10,
            axislabel_opts=opts.LabelOpts(
                font_size=12,
                font_family='Time New Roman',
                font_weight='normal',
            ),
            axistick_opts=opts.AxisTickOpts(
                is_inside=True, length=5,
            ),
            axisline_opts=opts.AxisLineOpts(
                is_show=True, linestyle_opts=opts.LineStyleOpts(width=2),
            ),
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
        polar_opts=opts.PolarOpts(),
        splitline_opt=opts.SplitLineOpts(is_show=False),
    )
        .add(
        series_name="0.5s（Original）",
        data=v1,
        linestyle_opts=opts.LineStyleOpts(color="#CD0000", width=2),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.0),
        color='#CD0000',
        symbol='triangle'
    )
        .add(
        series_name="1.0s（Augmented）",
        data=v2,
        linestyle_opts=opts.LineStyleOpts(color="#5CACEE", width=2),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.0),
        color='#5CACEE',
        symbol='rect'
    )
        .add(
        series_name="1.0s（Original）",
        data=v3,
        linestyle_opts=opts.LineStyleOpts(color="#35ad6b", width=2),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.0),
        color='#35ad6b',
        symbol='pin'
    )
        .add(
        series_name="2.0s（Augmented）",
        data=v4,
        linestyle_opts=opts.LineStyleOpts(color="#aa23ff", width=2),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.0),
        color='#aa23ff',
        symbol='roundRect'
    )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
        legend_opts=opts.LegendOpts(item_height=21, item_width=40, textstyle_opts=opts.TextStyleOpts(color="#000000",
                               font_size=15, font_weight='normal', font_family='Time New Roman')),
    )
        .render("Model_Aug_Comp.html")
)
