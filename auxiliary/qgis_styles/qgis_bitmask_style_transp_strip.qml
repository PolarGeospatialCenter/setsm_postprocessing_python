<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.18.1-Zürich" maxScale="0" minScale="1e+08" hasScaleBasedVisibilityFlag="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal mode="0" enabled="0" fetchMode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <property key="WMSBackgroundLayer" value="false"/>
    <property key="WMSPublishDataSourceUrl" value="false"/>
    <property key="embeddedWidgets/count" value="0"/>
    <property key="identify/format" value="Value"/>
  </customproperties>
  <pipe>
    <provider>
      <resampling zoomedOutResamplingMethod="nearestNeighbour" enabled="false" zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2"/>
    </provider>
    <rasterrenderer alphaBand="-1" type="paletted" nodataColor="" opacity="1" band="1">
      <rasterTransparency>
        <singleValuePixelList>
          <pixelListEntry max="2" percentTransparent="60" min="1"/>
          <pixelListEntry max="3" percentTransparent="45" min="3"/>
          <pixelListEntry max="4" percentTransparent="60" min="4"/>
          <pixelListEntry max="6" percentTransparent="45" min="5"/>
          <pixelListEntry max="7" percentTransparent="40" min="7"/>
        </singleValuePixelList>
      </rasterTransparency>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry value="0" label="0: &quot;good data&quot;" alpha="0" color="#ffffff"/>
        <paletteEntry value="1" label="1: bad edge data" alpha="0" color="#ebe76c"/>
        <paletteEntry value="2" label="2: water" alpha="255" color="#009ac4"/>
        <paletteEntry value="3" label="3: water and edge" alpha="255" color="#009ac4"/>
        <paletteEntry value="4" label="4: cloud" alpha="255" color="#ff4747"/>
        <paletteEntry value="5" label="5: cloud and edge" alpha="255" color="#ff4747"/>
        <paletteEntry value="6" label="6: cloud and water" alpha="255" color="#885bf0"/>
        <paletteEntry value="7" label="7: cloud, water, and edge" alpha="255" color="#885bf0"/>
      </colorPalette>
      <colorramp type="randomcolors" name="[source]">
        <Option/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast gamma="1" brightness="0" contrast="0"/>
    <huesaturation colorizeBlue="128" saturation="0" colorizeOn="0" grayscaleMode="0" colorizeRed="255" colorizeStrength="100" colorizeGreen="128"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
