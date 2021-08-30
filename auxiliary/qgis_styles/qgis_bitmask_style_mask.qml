<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis maxScale="0" version="3.6.3-Noosa" hasScaleBasedVisibilityFlag="0" minScale="1e+08" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <customproperties>
    <property value="false" key="WMSBackgroundLayer"/>
    <property value="false" key="WMSPublishDataSourceUrl"/>
    <property value="0" key="embeddedWidgets/count"/>
    <property value="Value" key="identify/format"/>
  </customproperties>
  <pipe>
    <rasterrenderer alphaBand="-1" band="1" opacity="1" type="paletted">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry value="0" color="#ffffff" label="0: &quot;good data&quot;" alpha="0"/>
        <paletteEntry value="1" color="#ebe76c" label="1: bad edge data" alpha="255"/>
        <paletteEntry value="2" color="#009ac4" label="2: water" alpha="255"/>
        <paletteEntry value="3" color="#0bce66" label="3: water and edge" alpha="255"/>
        <paletteEntry value="4" color="#ff4747" label="4: cloud" alpha="255"/>
        <paletteEntry value="5" color="#e29a3c" label="5: cloud and edge" alpha="255"/>
        <paletteEntry value="6" color="#885bf0" label="6: cloud and water" alpha="255"/>
        <paletteEntry value="7" color="#9a572e" label="7: cloud, water, and edge" alpha="255"/>
      </colorPalette>
      <colorramp name="[source]" type="randomcolors"/>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeBlue="128" saturation="0" colorizeGreen="128" colorizeStrength="100" colorizeRed="255" colorizeOn="0" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
