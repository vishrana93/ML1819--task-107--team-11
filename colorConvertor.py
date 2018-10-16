import colorsys

hexColors = []
with open('link_color.csv', 'r') as f:
    for line in f:
        for color in line.split():
            hexColors.append(color)

# convert colors from hexadecimal to hsv and print to file hsvColors.csv
with open('hsvlink_color.csv', 'w') as h:
    for color in hexColors:
        # catch any entries equal to 0
        try:
            color = list(int(color[i:i+2], 16) for i in (0, 2 ,4))
        except ValueError:
            color = [0, 0, 0]

        # must divide by 255 as co-ordinates are in range 0 to 1
        hsv = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        # rescale to hsv
        x = round(hsv[0]*360, 1)
        y = round(hsv[1]*100, 1)
        z = round(hsv[2]*100, 1)
        h.write("%s %s %s\n" % (x, y, z))



# converts colours from hexadecimal to rgb
# h = 'C0DEED'
# rgb = list(int(h[i:i+2], 16) for i in (0, 2 ,4))
# print ("this is original rgb", rgb)

# NOTE: tuples are immutable
# a = rgb[0]/255
# b = rgb[1]/255
# c = rgb[2]/255
#
# # NOTE: must divide by 255 as co-ordinates are in range 0 to 1.
# hsv = colorsys.rgb_to_hsv(a, b, c)
# # print ("This is hsv: ", hsv)
# x = round(hsv[0]*360, 1)
# y = round(hsv[1]*100, 1)
# z = round(hsv[2]*100, 1)
# print ("HSV: ", x, y, z)
