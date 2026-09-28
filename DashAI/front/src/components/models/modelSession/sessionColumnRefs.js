/**
 * Synthetic-key representation of a session's ColumnRef (see the backend's
 * DashAI.back.preprocessing.column_ref module). A raw dataset column is
 * just its own name; a converter's not-yet-materialized output group is
 * represented as a synthetic string key, so every UI piece that already
 * knows how to work with a flat `{name: string}` column list (Autocomplete
 * options, MaterialReactTable rows, columnTypes maps) can represent a group
 * without knowing groups exist at all.
 *
 * A step's output isn't always one homogeneous type — e.g. SimpleImputer
 * with "most_frequent"/"constant" just preserves each scope column's own
 * type, so a scope mixing a categorical and a numeric column produces both
 * kinds of columns. `slot` (a DashAI type's display_name(), e.g.
 * "Categorical") picks out just the columns of one declared type from a
 * step's output, mirroring the backend's GroupColumnRef.slot /
 * SessionPreprocessor.resolved_slots. A step with only one declared type —
 * the common case, and everything before slots existed — has no slot
 * (`slot: null`), which means "the whole group," unchanged from before.
 */

const GROUP_KEY_PREFIX = "__group__";
const SLOT_SEPARATOR = "__slot__";
const NAME_SEPARATOR = "__name__";
// The step, then at most one of a slot or a column name. Only the first
// separator after the step counts, so a slot or column name that itself
// contains a separator stays intact.
const GROUP_KEY_PATTERN = new RegExp(
  `^${GROUP_KEY_PREFIX}(\\d+)(?:${SLOT_SEPARATOR}(.*)|${NAME_SEPARATOR}(.*))?$`,
  "s",
);

export const groupKey = (step, slot = null) =>
  slot == null
    ? `${GROUP_KEY_PREFIX}${step}`
    : `${GROUP_KEY_PREFIX}${step}${SLOT_SEPARATOR}${slot}`;

// A generated column whose name is known before fit (e.g. DateFeatures'
// "date_month"), mirroring the backend's GroupColumnRef.name.
export const namedGroupKey = (step, name) =>
  `${GROUP_KEY_PREFIX}${step}${NAME_SEPARATOR}${name}`;

export const isGroupKey = (key) =>
  typeof key === "string" && key.startsWith(GROUP_KEY_PREFIX);

const parseGroupKey = (key) => {
  const match = GROUP_KEY_PATTERN.exec(key);
  return {
    step: Number(match[1]),
    slot: match[2] ?? null,
    name: match[3] ?? null,
  };
};

export const stepFromGroupKey = (key) => parseGroupKey(key).step;

export const slotFromGroupKey = (key) => parseGroupKey(key).slot;

/** ColumnRef -> synthetic key */
export const refToKey = (ref) => {
  if (ref.kind === "raw") return ref.name;
  if (ref.name != null) return namedGroupKey(ref.step, ref.name);
  return groupKey(ref.step, ref.slot ?? null);
};

/** synthetic key -> ColumnRef */
export const keyToRef = (key) => {
  if (!isGroupKey(key)) return { kind: "raw", name: key };
  const { step, slot, name } = parseGroupKey(key);
  if (name != null) return { kind: "group", step, name };
  return slot == null ? { kind: "group", step } : { kind: "group", step, slot };
};

/**
 * The ColumnRef that points at an item of the estimated dataset structure
 * (see the backend's infer_structure): an original column by name, a
 * generated column by its step and name, a block by its step and slot (a
 * lone block has no slot: it is its step's whole group).
 */
export const itemToRef = (item) => {
  if (item.kind === "column") {
    return item.origin == null
      ? { kind: "raw", name: item.name }
      : { kind: "group", step: item.origin, name: item.name };
  }
  return item.slot == null
    ? { kind: "group", step: item.step }
    : { kind: "group", step: item.step, slot: item.slot };
};

const blockLabel = (block, stepName) => {
  const base =
    block.label && block.label !== "output"
      ? `${stepName}: ${block.label}`
      : `${stepName}: output`;
  // "N" when the column count is only known after fit.
  return `${base} (${block.count ?? "N"})`;
};

/**
 * Selector options for the items of an estimated dataset state: one key per
 * item (the synthetic key of the ColumnRef pointing at it), its type, and a
 * label for every item that is not an original column. Plugs into the same
 * `{allKeys, columnTypes, optionLabels}` shape ColumnSelector and
 * DivideDatasetColumns already take.
 *
 * @param {Array<object>} state items from a StructureResult state
 * @param {string[]} stepDisplayNames one name per step (see
 *   buildStepDisplayNames)
 */
export function stateToOptions(state, stepDisplayNames = []) {
  const allKeys = [];
  const columnTypes = {};
  const optionLabels = {};
  (state || []).forEach((item) => {
    const key = refToKey(itemToRef(item));
    allKeys.push(key);
    columnTypes[key] = { type: item.type ?? null, dtype: item.dtype ?? null };
    if (item.kind === "block") {
      optionLabels[key] = blockLabel(item, stepDisplayNames[item.step] ?? "");
    } else if (item.origin != null) {
      optionLabels[key] = item.name;
    }
  });
  return { allKeys, columnTypes, optionLabels };
}

/**
 * A label for a ColumnRef without the estimated structure, for sessions
 * that are already created (e.g. the session info panel).
 */
export function labelForRef(ref, stepDisplayNames = []) {
  if (ref.kind === "raw") return ref.name;
  if (ref.name != null) return ref.name;
  const stepName = stepDisplayNames[ref.step] ?? `${ref.step}`;
  return ref.slot == null
    ? `${stepName}: output`
    : `${stepName}: output (${ref.slot})`;
}

// A step's declared output, normalized to a list of slots — even a
// homogeneous step (the common case) is one "slot" with slot: null, so
// every caller iterates the same shape regardless of how many there are.
// Falls back defensively for a step that predates outputSlots.
function stepOutputSlots(step) {
  if (Array.isArray(step?.outputSlots) && step.outputSlots.length > 0) {
    return step.outputSlots;
  }
  return [
    {
      slot: null,
      type: step?.outputType ?? null,
      dtype: step?.outputDtype ?? null,
    },
  ];
}

/**
 * One display name per step, disambiguated when the sequence has more than
 * one step of the same converter type (same registry name, or same
 * fallback string when `convertersMeta` hasn't loaded yet) — the first
 * occurrence keeps the bare name, later ones get " (2)", " (3)", etc., by
 * order of appearance. Computed over the FULL `steps` array regardless of
 * any later truncation (e.g. buildColumnKeysAndTypes's `uptoStep`), so a
 * step's numbering never shifts depending on which view is asking.
 */
export function buildStepDisplayNames(steps, convertersMeta = {}) {
  const baseNames = (steps || []).map(
    (step) => convertersMeta[step?.converter]?.display_name || step?.converter,
  );

  const totalByName = {};
  baseNames.forEach((name) => {
    totalByName[name] = (totalByName[name] || 0) + 1;
  });

  const seenSoFar = {};
  return baseNames.map((name) => {
    if (totalByName[name] <= 1) return name;
    seenSoFar[name] = (seenSoFar[name] || 0) + 1;
    return seenSoFar[name] === 1 ? name : `${name} (${seenSoFar[name]})`;
  });
}

/**
 * Every column key a session's preprocessing sequence can be scoped over,
 * up to (and not including) `uptoStep`: every raw dataset column, plus one
 * group key per declared slot of every converter step before it (usually
 * one key per step; more than one only when that step's scope mixed
 * column types). Passing no `uptoStep` includes every configured step
 * (used once a sequence is final and being displayed, e.g. in
 * SelectColumnsStep, where every step is already "before" the
 * column-selection step that comes after all of them).
 *
 * `convertersMeta` (registry name -> component, e.g. from getComponents)
 * is optional: when supplied, option labels use each step's disambiguated
 * display name (see buildStepDisplayNames) instead of its raw registry
 * name.
 */
export function buildColumnKeysAndTypes({
  datasetTypes,
  preprocessing,
  uptoStep,
  convertersMeta = {},
}) {
  const steps = preprocessing || [];
  const limit = uptoStep === undefined ? steps.length : uptoStep;
  const displayNames = buildStepDisplayNames(steps, convertersMeta);

  const columnTypes = { ...datasetTypes };
  const optionLabels = {};
  const allKeys = Object.keys(datasetTypes || {});

  for (let index = 0; index < limit; index += 1) {
    const step = steps[index];
    const name = displayNames[index];
    stepOutputSlots(step).forEach(({ slot, type, dtype }) => {
      const key = groupKey(index, slot);
      columnTypes[key] = { type: type || null, dtype: dtype || null };
      optionLabels[key] = slot
        ? `${name}: output (${slot})`
        : `${name}: output`;
      allKeys.push(key);
    });
  }

  return { allKeys, columnTypes, optionLabels };
}

/**
 * Every RAW dataset column name needed to compute a list of ColumnRef,
 * walking group refs back to their step's own scope recursively (a group
 * ref's step may itself reference an earlier group, chained arbitrarily
 * deep). This is what a caller must actually supply values for — e.g.
 * manual prediction, where the backend only ever accepts real dataset
 * columns as input (see BaseTask.process_manual_input), never a
 * converter's resolved output name like "pca_1": it runs the raw values
 * through the session's persisted preprocessor itself before predicting.
 */
export function rawColumnsNeededFor(refs, steps) {
  const needed = [];
  const seen = new Set();
  const visit = (ref) => {
    if (ref.kind === "raw") {
      if (!seen.has(ref.name)) {
        seen.add(ref.name);
        needed.push(ref.name);
      }
      return;
    }
    const step = (steps || [])[ref.step];
    if (!step) return;
    (step.scope || []).forEach(visit);
  };
  (refs || []).forEach(visit);
  return needed;
}
