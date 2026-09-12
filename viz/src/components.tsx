import * as d3 from 'd3'
import type { ReactNode } from 'react'

export function Panel({
  title,
  blurb,
  formula,
  children,
}: {
  title: string
  blurb: string
  formula: string
  children: ReactNode
}) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      <p className="blurb">{blurb}</p>
      <pre className="formula">{formula}</pre>
      {children}
    </section>
  )
}

/** Vertical bars via divs so CSS transitions animate height changes. */
export function Bars({
  values,
  labels,
  clipped,
  height = 160,
  colors,
}: {
  values: number[]
  labels?: string[]
  clipped?: boolean[]
  height?: number
  colors?: (v: number, i: number) => string
}) {
  const lo = Math.min(0, ...values)
  const hi = Math.max(1e-9, ...values)
  const y = d3.scaleLinear().domain([lo, hi]).range([height, 0])
  const zero = y(0)
  return (
    <div className="bars" style={{ height }}>
      {values.map((v, i) => {
        const top = Math.min(y(v), zero)
        const h = Math.abs(y(v) - zero)
        return (
          <div key={i} className="bar-wrap">
            <div
              className={'bar' + (clipped?.[i] ? ' clipped' : '')}
              style={{
                top,
                height: h,
                background: colors ? colors(v, i) : '#4a7abf',
                transition: 'top .5s, height .5s',
              }}
              title={v.toFixed(4)}
            />
            <div className="bar-label">{labels?.[i] ?? i}</div>
            <div className="bar-value">{v.toFixed(2)}</div>
          </div>
        )
      })}
      <div className="axis" style={{ top: zero }} />
    </div>
  )
}

export function LineChart({
  series,
  band,
  width = 480,
  height = 140,
  yLabel,
}: {
  series: number[]
  band?: { lo: number[]; hi: number[] }
  width?: number
  height?: number
  yLabel?: string
}) {
  if (series.length === 0) return null
  const all = band ? [...band.lo, ...band.hi] : series
  const x = d3
    .scaleLinear()
    .domain([0, Math.max(1, series.length - 1)])
    .range([30, width - 8])
  const y = d3
    .scaleLinear()
    .domain([Math.min(...all), Math.max(...all) || 1e-9])
    .nice()
    .range([height - 20, 8])
  const line = d3
    .line<number>()
    .x((_, i) => x(i))
    .y((v) => y(v))
  const area = band
    ? d3
        .area<number>()
        .x((_, i) => x(i))
        .y0((_, i) => y(band.lo[i]))
        .y1((_, i) => y(band.hi[i]))(series.map((_, i) => i) as number[])
    : null
  const ticks = y.ticks(4)
  return (
    <svg width={width} height={height} className="linechart">
      {area && <path d={area} fill="#4a7abf" opacity={0.15} />}
      <path d={line(series) ?? ''} fill="none" stroke="#4a7abf" strokeWidth={2} />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={30} x2={width - 8} y1={y(t)} y2={y(t)} stroke="#ddd" />
          <text x={2} y={y(t) + 3} fontSize={9} fill="#888">
            {t}
          </text>
        </g>
      ))}
      {yLabel && (
        <text x={2} y={10} fontSize={9} fill="#888">
          {yLabel}
        </text>
      )}
    </svg>
  )
}
