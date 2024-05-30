import vvdutils as vvd
import folium

# https://blog.csdn.net/ThsPool/article/details/132769692
# https://cloud.tencent.com/developer/article/1971955

if __name__ == "__main__":

    min_lon, max_lon = 70, 138
    min_lat, max_lat = 17.6, 53.8

    m =  folium.Map(
        control_scale=True,
        zoom_control=True,
        location=[36, 105],
        zoom_start=5,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        )

    yiweizhang_lat = 121.246259
    yiweizhang_lon = 31.177461

    folium.LayerControl().add_to(m)

    icon_boy_image = 'boy.png'
    icon_girl_image = 'girl.png'
    shadow_image = 'shadow.png'

    # icon = folium.CustomIcon(
    #     icon_image,
    #     icon_size=(40, 70)
    # )

    icon_boy = folium.CustomIcon(
        icon_boy_image,
        icon_size=(40, 64),
        icon_anchor=(22, 94),
        shadow_image=shadow_image,
        shadow_size=(55, 34),
        shadow_anchor=(9, 65)
    )

    icon_girl = folium.CustomIcon(
        icon_girl_image,
        icon_size=(40, 64),
        icon_anchor=(22, 94),
        shadow_image=shadow_image,
        shadow_size=(55, 34),
        shadow_anchor=(9, 65)
    )


    folium.Marker(
        location=[yiweizhang_lon+2, yiweizhang_lat-5], icon=icon_boy, popup="Name: boy \nTel: 18512341234"
    ).add_to(m)

    folium.Marker(
        location=[yiweizhang_lon-2, yiweizhang_lat-5], icon=icon_girl, popup="Name: girl \nTel: 18598765432"
    ).add_to(m)

    m.save('show_map.html')
    pass